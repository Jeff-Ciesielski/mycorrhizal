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
from enum import auto, Enum

from mycorrhizal.enoki.core import TimeoutMessage

# Import the Enoki modules
from mycorrhizal.enoki import (
    State,
    StateMachine,
    StateConfiguration,
    SharedContext,
    StateRef,
    Push,
    Pop,
    Retry,
    StateMachineComplete,
    LabeledTransition,
    PopFromEmptyStack,
    ValidationError,
    DefaultStates
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

    class TestTransitions(Enum):
        GO_TO_B = auto()
        POP_BACK = auto()

    class StateA(State):
        CONFIG = StateConfiguration()

        def transitions(cls):
            return [
                LabeledTransition(
                    TestStringResolution.TestTransitions.GO_TO_B,
                    StateRef("test_enoki.TestStringResolution.StateB"),
                ),
            ]

        def on_state(cls, ctx: SharedContext):
            return TestStringResolution.TestTransitions.GO_TO_B

    class StateB(State):
        CONFIG = StateConfiguration()

        def transitions(cls):
            return [
                LabeledTransition(TestStringResolution.TestTransitions.POP_BACK, Pop)
            ]

        def on_state(cls, ctx: SharedContext):
            return TestStringResolution.TestTransitions.POP_BACK

    def test_transition_to_string_state_reference(self):
        """States should be able to transition via string references."""

        result = run_fsm_scenario(TestStringResolution.StateA, timeout=0)
        self.assertEqual(result.get("final_state"), TestStringResolution.StateB)


class TestBasicStateTransitions(EnokiTestCase):
    def test_unhandled_returns_unhandled(self):
        """Returning None should result in unhandled transition."""

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertNoTransition(TestState)

    def test_push_single_state(self):
        """Push with single state should work correctly."""

        class TestTransitions(Enum):
            PUSH_B = auto()
            POP_BACK = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [
                    LabeledTransition(TestTransitions.PUSH_B, Push(StateB)),
                ]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_B

        class StateB(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        result = run_state_on_state(StateA)
        self.assertEqual(result, TestTransitions.PUSH_B)

        # Verify the transition mapping
        transitions = StateA.transitions()
        # Should be a list of LabeledTransition
        push_targets = [
            t
            for t in transitions
            if isinstance(t, LabeledTransition) and t.label == TestTransitions.PUSH_B
        ]
        self.assertTrue(push_targets)
        self.assertIsInstance(push_targets[0].transition, Push)

    def test_push_multiple_states(self):
        """Push with multiple states should work correctly."""

        class TestTransitions(Enum):
            PUSH_MULTI = auto()
            POP_BACK = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [
                    LabeledTransition(TestTransitions.PUSH_MULTI, Push(StateB, StateC)),
                ]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_MULTI

        class StateB(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        class StateC(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []  # No transitions, terminal or will pop

            def on_state(cls, ctx: SharedContext):
                return None

        result = run_state_on_state(StateA)
        self.assertEqual(result, TestTransitions.PUSH_MULTI)

        # Verify the transition mapping
        transitions = StateA.transitions()
        push_targets = [
            t
            for t in transitions
            if isinstance(t, LabeledTransition)
            and t.label == TestTransitions.PUSH_MULTI
        ]
        self.assertTrue(push_targets)
        self.assertIsInstance(push_targets[0].transition, Push)
        self.assertEqual(len(push_targets[0].transition.push_states), 2)

    def test_pop_transition(self):
        """Pop transitions should work correctly."""

        class TestTransitions(Enum):
            POP_BACK = auto()

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        result = run_state_on_state(TestState)
        self.assertEqual(result, TestTransitions.POP_BACK)

        # Verify the transition mapping
        transitions = TestState.transitions()
        pop_targets = [
            t
            for t in transitions
            if isinstance(t, LabeledTransition) and t.label == TestTransitions.POP_BACK
        ]
        self.assertTrue(pop_targets)
        self.assertIs(pop_targets[0].transition, Pop)


class TestStateLifecycle(EnokiTestCase):
    """Test state lifecycle methods in  architecture."""

    def test_on_enter_called(self):
        """on_enter should be callable and not crash."""

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []

            def on_enter(cls, ctx: SharedContext):
                ctx.common.entered = True

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertLifecycleMethodCalled(TestState, "on_enter")

    def test_on_leave_called(self):
        """on_leave should be callable and not crash."""

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []

            def on_leave(cls, ctx: SharedContext):
                ctx.common.left = True

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertLifecycleMethodCalled(TestState, "on_leave")

    def test_on_fail_called_on_retry_limit(self):
        """on_fail should be callable and handle retry exhaustion."""

        class TestTransitions(Enum):
            FAILED = auto()

        class TestState(State):
            CONFIG = StateConfiguration(retries=3, timeout=0.1)

            def transitions(cls):
                return [LabeledTransition(TestTransitions.FAILED, ErrorState), Retry]

            def on_fail(cls, ctx: SharedContext):
                ctx.common['failed'] = True
                return TestTransitions.FAILED

            def on_state(cls, ctx: SharedContext):
                return None
            
            def on_timeout(cls, ctx):
                return Retry

        class ErrorState(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        result = run_fsm_scenario(TestState)
        self.assertEqual(result['final_state'], ErrorState)
        self.assertTrue(result['ctx'].common.get('failed'))

class TestStateConfiguration(EnokiTestCase):
    """Test StateConfiguration and state properties."""

    def test_timeout_configuration(self):
        """States should properly configure timeouts."""

        class TimeoutState(State):
            CONFIG = StateConfiguration(timeout=10.0)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertStateConfiguration(TimeoutState, timeout=10.0)

    def test_retry_configuration(self):
        """States should properly configure retry limits."""

        class RetryState(State):
            CONFIG = StateConfiguration(retries=5)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertStateConfiguration(RetryState, retries=5)

    def test_terminal_configuration(self):
        """States should properly configure terminal status."""

        class TerminalState(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertStateConfiguration(TerminalState, terminal=True)

    def test_can_dwell_configuration(self):
        """States should properly configure dwelling capability."""

        class DwellState(State):
            CONFIG = StateConfiguration(can_dwell=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertStateConfiguration(DwellState, can_dwell=True)


class TestSharedContext(EnokiTestCase):
    """Test SharedContext functionality."""

    def test_shared_state_modification(self):
        """States should be able to modify shared state."""

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                ctx.common.test_value = 42
                return None

        self.assertSharedStateModified(TestState, "test_value", 42)

    def test_message_handling(self):
        """States should be able to access and respond to messages."""

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("action") == "test":
                    ctx.common.received_test = True
                return None

        context = make_mock_context(msg={"action": "test"})
        run_state_on_state(TestState, context)
        self.assertTrue(context.common.received_test)

    def test_send_message_functionality(self):
        """States should be able to send messages."""

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                ctx.send_message({"response": "sent"})
                return None

        context = make_mock_context()
        run_state_on_state(TestState, context)

        # Verify send_message was called
        context.send_message.assert_called_once_with({"response": "sent"})

    def test_log_functionality(self):
        """States should be able to log messages."""

        class TestState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                ctx.log("Test log message")
                return None

        context = make_mock_context()
        run_state_on_state(TestState, context)

        # Verify log was called
        context.log.assert_called_once_with("Test log message")


class TestStateMachineLifecycle(StateMachineTestCase):
    """Test StateMachine lifecycle and behavior."""

    def test_fsm_starts_in_initial_state(self):
        """FSM should start in the specified initial state."""

        class TestTransitions(Enum):
            POP_BACK = auto()

        class InitialState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        sm = StateMachine(InitialState)
        self.assertEqual(sm.current_state, InitialState)

    def test_fsm_transitions_between_states(self):
        """FSM should transition between states correctly."""

        class TestTransitions(Enum):
            GO_TO_B = auto()
            FINAL = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.GO_TO_B, StateB)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_B

        class StateB(State):
            CONFIG = StateConfiguration(can_dwell=True)  # Prevent blocking

            def transitions(cls):
                return [LabeledTransition(TestTransitions.FINAL, Done)]

            def on_state(cls, ctx: SharedContext):
                return None

        class Done(State):
            CONFIG = StateConfiguration(terminal=True)
            
            def on_state(cls, ctx):
                pass            

        sm = StateMachine(StateA)
        sm.tick()

        self.assertEqual(sm.current_state, StateB)

    def test_fsm_completes_on_terminal_state(self):
        """FSM should complete when reaching a terminal state."""
        
        class InitialState(State):
            def transitions(cls):
                return TerminalState
            
            def on_state(cls, ctx):
                return TerminalState

        class TerminalState(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertFSMCompletes(InitialState)

    def test_fsm_blocks_in_untimed_state(self):
        """FSM should raise BlockedInUntimedState for non-dwelling states."""

        class TestTransitions(Enum):
            NADA = auto()

        class BlockingState(State):
            CONFIG = StateConfiguration()  # No timeout, no dwelling

            def transitions(cls):
                return [LabeledTransition(TestTransitions.NADA, None)]

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertFSMBlocks(BlockingState)

    def test_fsm_reset_functionality(self):
        """FSM reset should return to initial state and clear stack."""

        class TestTransitions(Enum):
            PUSH_B = auto()
            POP_BACK = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.PUSH_B, Push(StateB, Done))]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_B

        class StateB(State):
            CONFIG = StateConfiguration(can_dwell=True)

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK
            
        class Done(State):
            CONFIG = StateConfiguration(terminal=True)
            
            def on_state(cls, ctx):
                pass

        print(StateA.name)
        sm = StateMachine(StateA)
        with self.assertRaises(StateMachineComplete):
            sm.tick()  # Should transition to StateB, then Done

        # Verify state changed
        self.assertEqual(sm.current_state, Done)

        # Reset and verify
        sm.reset()
        self.assertEqual(sm.current_state, StateA)
        self.assertEqual(len(sm.state_stack), 0)


class TestStackOperations(StateMachineTestCase):
    """Test push/pop stack operations."""

    def test_push_multiple_states_uses_stack(self):
        """Pushing multiple states should use the state stack correctly."""

        class TestTransitions(Enum):
            PUSH_MULTI = auto()
            POP_BACK = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [
                    LabeledTransition(TestTransitions.PUSH_MULTI, Push(StateB, StateC))
                ]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_MULTI

        class StateB(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                ctx.common["B_entered"] = True
                return TestTransitions.POP_BACK

        class StateC(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                ctx.common["C_entered"] = True
                return None

        cd = {"B_entered": False, "C_entered": False}
        sm = StateMachine(StateA, common_data=cd)

        # First tick: A -> B, push C
        with self.assertRaises(StateMachineComplete):
            sm.tick()
        self.assertEqual(sm.current_state, StateC)
        self.assertEqual(len(sm.state_stack), 0)
        self.assertTrue(cd["B_entered"])
        self.assertTrue(cd["C_entered"])

    def test_pop_from_empty_stack_handled(self):
        """Popping from empty stack should be handled gracefully."""

        class TestTransitions(Enum):
            POP_BACK = auto()

        class PopState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        sm = StateMachine(PopState)

        # This should handle the empty stack gracefully
        # The exact behavior depends on implementation
        with self.assertRaises(PopFromEmptyStack):
            sm.tick()


class TestValidation(StateMachineTestCase):
    """Test state machine validation."""
    
    def test_invalid_timeout(self):
        class TestState(State):
            CONFIG = StateConfiguration(can_dwell=True, terminal=True)
            
            def on_state(cls, ctx):
                pass

        sm = StateMachine(TestState, DefaultStates.Error)
        with self.assertRaises(ValueError):
            sm.tick(timeout="invalid value")
            

    def test_missing_state_reference_validation(self):
        """StateMachine should validate state references during construction."""

        class TestTransitions(Enum):
            BAD = auto()

        class BadState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [
                    LabeledTransition(
                        TestTransitions.BAD, StateRef("nonexistent.module.FakeState")
                    )
                ]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.BAD

        self.assertValidationError(BadState)

    def test_invalid_transition_enum_validation(self):
        """StateMachine should validate transition enums."""

        class BadState(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                # Intentionally use a string label to trigger validation error
                return [LabeledTransition("not_an_enum", StateRef("some_state"))]

            def on_state(cls, ctx: SharedContext):
                return None

        self.assertValidationError(BadState)

    def test_timeout_without_handler_warning(self):
        """States with timeout but no timeout handler should generate warnings."""

        class TimeoutTransitions(Enum):
            UNHANDLED = auto()

        class TimeoutState(State):
            CONFIG = StateConfiguration(timeout=5.0)

            def transitions(cls):
                # No timeout handler, only unhandled
                return [LabeledTransition(TimeoutTransitions.UNHANDLED, None)]

            def on_state(cls, ctx: SharedContext):
                return None

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

        class TimeoutTransitions(Enum):
            RETURN_TO_IDLE = auto()
            GO_TO_LONG_STATE = auto()

        class Idle(State):
            CONFIG = StateConfiguration(can_dwell=True)

            def transitions(cls):
                return [
                    LabeledTransition(TimeoutTransitions.GO_TO_LONG_STATE, LongState)
                ]

            def on_state(cls, ctx):
                return TimeoutTransitions.GO_TO_LONG_STATE

        class LongState(State):
            CONFIG = StateConfiguration(timeout=0.2)

            def transitions(cls):
                return [LabeledTransition(TimeoutTransitions.RETURN_TO_IDLE, Idle)]

            def on_state(cls, ctx):
                return None

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

        class ImagingTransitions(Enum):
            LIGHTING_SET = auto()
            PICTURE_TAKEN = auto()
            FAILED = auto()
            TIMEOUT = auto()

        class SetImageLighting(State):
            CONFIG = StateConfiguration(timeout=10.0, retries=3)

            def transitions(cls):
                return [
                    LabeledTransition(ImagingTransitions.LIGHTING_SET, TakePicture),
                    LabeledTransition(ImagingTransitions.FAILED, ErrorState),
                    LabeledTransition(ImagingTransitions.TIMEOUT, Retry),
                ]

            def on_enter(cls, ctx: SharedContext):
                ctx.log("Setting up lighting...")
                # Simulate async setup
                spawn(cls._simulate_setup, ctx)

            def _simulate_setup(cls, ctx: SharedContext):
                sleep(0.1)  # Short simulation
                ctx.send_message({"type": "lighting_ready"})

            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("type") == "lighting_ready":
                    return ImagingTransitions.LIGHTING_SET
                return None

            def on_timeout(cls, ctx: SharedContext):
                ctx.log("Lighting setup timed out")
                return Retry

            def on_fail(cls, ctx: SharedContext):
                ctx.log("Lighting setup failed after all retries")
                return ImagingTransitions.FAILED

        class TakePicture(State):
            CONFIG = StateConfiguration(timeout=15.0, retries=2)

            def transitions(cls):
                return [
                    LabeledTransition(
                        ImagingTransitions.PICTURE_TAKEN, ProcessingComplete
                    ),
                    LabeledTransition(ImagingTransitions.FAILED, ErrorState),
                    LabeledTransition(ImagingTransitions.TIMEOUT, Retry),
                ]

            def on_enter(cls, ctx: SharedContext):
                ctx.log("Taking picture...")
                spawn(cls._simulate_capture, ctx)

            def _simulate_capture(cls, ctx: SharedContext):
                sleep(0.1)  # Short simulation
                ctx.send_message({"type": "picture_complete", "success": True})

            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("type") == "picture_complete":
                    if ctx.msg.get("success"):
                        return ImagingTransitions.PICTURE_TAKEN
                    else:
                        return ImagingTransitions.FAILED
                return None

        class ProcessingComplete(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                ctx.log("Processing complete!")
                return None

        class ErrorState(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                ctx.log("Error state reached")
                return None

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

        class MenuTransitions(Enum):
            ENTER_SUBMENU = auto()
            BACK = auto()
            EXIT = auto()

        class MainMenu(State):
            CONFIG = StateConfiguration(can_dwell=True)

            def transitions(cls):
                return [
                    LabeledTransition(
                        MenuTransitions.ENTER_SUBMENU, Push(SubMenu, MainMenu)
                    ),
                    LabeledTransition(MenuTransitions.EXIT, ProcessingComplete),
                ]

            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("action") == "enter_submenu":
                    return MenuTransitions.ENTER_SUBMENU
                elif ctx.msg and ctx.msg.get("action") == "exit":
                    return MenuTransitions.EXIT

        class SubMenu(State):
            CONFIG = StateConfiguration(can_dwell=True)

            def transitions(cls):
                return [LabeledTransition(MenuTransitions.BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                if ctx.msg and ctx.msg.get("action") == "back":
                    return MenuTransitions.BACK

        class ProcessingComplete(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        # Test the menu navigation
        messages = [{"action": "enter_submenu"}, {"action": "back"}, {"action": "exit"}]

        scenario_result = run_fsm_scenario(
            MainMenu, messages=messages, max_iterations=10, timeout=0
        )

        self.assertEqual(scenario_result["error"], None)
        # Should reach ProcessingComplete
        self.assertEqual(scenario_result["final_state"], ProcessingComplete)
        self.assertTrue(scenario_result["completed"])


class TestAnalysisAndVisualization(StateMachineTestCase):
    """Test analysis and visualization features."""

    def test_state_discovery(self):
        """StateMachine should discover all reachable states."""

        class TestTransitions(Enum):
            GO_TO_B = auto()
            GO_TO_C = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.GO_TO_B, StateB)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_B

        class StateB(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.GO_TO_C, StateC)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_C

        class StateC(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        sm = StateMachine(StateA)

        # Should discover all three states
        expected_states = [StateA, StateB, StateC]
        self.assertFSMAnalysis(sm, expected_states=expected_states)

    def test_state_grouping(self):
        """StateMachine should group states by base class."""

        class ImagingState(State):
            """Base class for imaging states."""

            pass

        class ImagingTransitions(Enum):
            NEXT = auto()
            PUSH = auto()
            POP = auto()

        class CaptureImage(ImagingState):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [
                    LabeledTransition(ImagingTransitions.NEXT, ProcessImage),
                    LabeledTransition(
                        ImagingTransitions.PUSH, Push(PushState, ProcessImage)
                    ),
                ]

            def on_state(cls, ctx: SharedContext):
                return ImagingTransitions.PUSH

        class PushState(ImagingState):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(ImagingTransitions.POP, Pop)]

            def on_state(cls, ctx):
                return ImagingTransitions.POP

        class ProcessImage(ImagingState):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        sm = StateMachine(CaptureImage)

        # Should group under ImagingState
        groups = sm.state_groups
        self.assertIn("ImagingState", groups)

    def test_push_pop_analysis(self):
        """StateMachine should analyze push/pop relationships."""

        class TestTransitions(Enum):
            PUSH_B = auto()
            POP_BACK = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.PUSH_B, Push(StateB, StateA))]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.PUSH_B

        class StateB(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP_BACK, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP_BACK

        sm = StateMachine(StateA)

        # Should analyze push/pop relationships
        relationships = sm.get_push_pop_relationships()

        # StateB should have push sources and pop targets
        stateb_name = StateB.name
        if stateb_name in relationships:
            stateb_info = relationships[stateb_name]
            self.assertTrue(stateb_info.get("is_pushed_to", False))
            self.assertTrue(stateb_info.get("has_pop", False))

    def test_mermaid_flowchart_generation(self):
        """StateMachine should generate Mermaid flowcharts."""

        class TestTransitions(Enum):
            GO_TO_C = auto()
            POP = auto()

        class StateA(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [
                    LabeledTransition(TestTransitions.GO_TO_C, Push(StateB, StateC))
                ]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.GO_TO_C

        class StateB(State):
            CONFIG = StateConfiguration()

            def transitions(cls):
                return [LabeledTransition(TestTransitions.POP, Pop)]

            def on_state(cls, ctx: SharedContext):
                return TestTransitions.POP

        class StateC(State):
            CONFIG = StateConfiguration(terminal=True)

            def transitions(cls):
                return []

            def on_state(cls, ctx: SharedContext):
                return None

        sm = StateMachine(StateA)

        # Should generate a flowchart
        flowchart = sm.generate_mermaid_flowchart()
        self.assertIsInstance(flowchart, str)
        self.assertIn("flowchart", flowchart.lower())
        self.assertIn("StateA", flowchart)
        self.assertIn("StateB", flowchart)
        self.assertIn("StateC", flowchart)


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
