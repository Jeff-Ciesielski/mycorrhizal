#!/usr/bin/env python3
"""
Enoki Testing Utilities

Testing infrastructure for the class-method-based Enoki architecture.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
import time
import gevent
from gevent import sleep

# Import the Enoki modules
from mycorrhizal.enoki import (
    State, StateMachine, TransitionEnum, GlobalTransitions,
    StateConfiguration, SharedContext, TimeoutMessage, Push, Pop,
    ValidationError, BlockedInUntimedState
)


def make_mock_context(
    common_data=None,
    msg=None,
    **kwargs
):
    """
    Create a mock SharedContext for testing state methods.
    
    Args:
        common_data: dict/object for shared state between states
        msg: message to set as current message
        **kwargs: additional attributes to set on the context
    
    Returns:
        SharedContext configured for testing
    """
    if common_data is None:
        common_data = SimpleNamespace()
    elif isinstance(common_data, dict):
        common_data = SimpleNamespace(**common_data)
    
    # Mock functions
    send_message_mock = Mock(return_value=None)
    log_mock = Mock(return_value=None)
    
    context = SharedContext(
        send_message=send_message_mock,
        log=log_mock,
        common=common_data,
        msg=msg,
        fsm=None  # Can be set by caller if needed
    )
    
    # Apply any additional kwargs
    for key, value in kwargs.items():
        setattr(context, key, value)
    
    return context


def run_state_method(state_class, method_name, context=None, **context_kwargs):
    """
    Run a specific state method (on_state, on_enter, etc.) with given context.
    
    Args:
        state_class: The state class to test
        method_name: Name of the method to call ('on_state', 'on_enter', etc.)
        context: SharedContext to pass, or None to create one
        **context_kwargs: Arguments for make_mock_context if context is None
    
    Returns:
        The result from the state method
    """
    if context is None:
        context = make_mock_context(**context_kwargs)
    
    method = getattr(state_class, method_name)
    return method(context)


def run_state_on_state(state_class, context=None, **context_kwargs):
    """
    Convenience function to run a state's on_state method.
    
    Args:
        state_class: The state class to test
        context: SharedContext to pass, or None to create one
        **context_kwargs: Arguments for make_mock_context if context is None
    
    Returns:
        The transition result from on_state
    """
    return run_state_method(state_class, 'on_state', context, **context_kwargs)


def simulate_state_lifecycle(state_class, messages=None, max_iterations=10, **context_kwargs):
    """
    Simulate a complete state lifecycle with multiple messages.
    
    Args:
        state_class: The state class to simulate
        messages: List of messages to send, or None for no messages
        max_iterations: Maximum iterations before giving up
        **context_kwargs: Arguments for make_mock_context
    
    Returns:
        List of (message, transition_result) tuples
    """
    if messages is None:
        messages = [None]  # Single iteration with no message
    
    context = make_mock_context(**context_kwargs)
    results = []
    
    # Call on_enter first
    try:
        state_class.on_enter(context)
    except Exception as e:
        results.append(('on_enter', e))
    
    # Process each message
    for msg in messages:
        context.msg = msg
        try:
            result = state_class.on_state(context)
            results.append((msg, result))
            
            # Break if we get a definitive transition
            if result != GlobalTransitions.UNHANDLED:
                break
                
        except Exception as e:
            results.append((msg, e))
            break
    
    # Call on_leave
    try:
        state_class.on_leave(context)
        results.append(('on_leave', None))
    except Exception as e:
        results.append(('on_leave', e))
    
    return results


class EnokiTestCase(unittest.TestCase):
    """Base test case class with assertion methods for Enoki FSM testing."""
    
    def assertStateTransition(self, state_class, expected_transition, 
                             context=None, **context_kwargs):
        """Assert that a state transitions to the expected transition type."""
        result = run_state_on_state(state_class, context, **context_kwargs)
        
        if isinstance(expected_transition, type) and issubclass(expected_transition, State):
            # Convert state class to string for comparison
            if isinstance(result, str):
                expected_name = expected_transition.name()
                self.assertEqual(result, expected_name, 
                               f"Expected transition to {expected_name}, got {result}")
            else:
                self.fail(f"Expected string state name, got {type(result)}: {result}")
        elif isinstance(expected_transition, TransitionEnum):
            self.assertEqual(result, expected_transition,
                           f"Expected {expected_transition}, got {result}")
        else:
            self.assertEqual(result, expected_transition)
    
    def assertNoTransition(self, state_class, context=None, **context_kwargs):
        """Assert that a state does not transition (returns UNHANDLED)."""
        result = run_state_on_state(state_class, context, **context_kwargs)
        self.assertEqual(result, GlobalTransitions.UNHANDLED,
                        f"Expected UNHANDLED, got {result}")
    
    def assertPushTransition(self, state_class, expected_states, 
                            context=None, **context_kwargs):
        """Assert that a state returns a Push transition with expected states."""
        result = run_state_on_state(state_class, context, **context_kwargs)
        self.assertIsInstance(result, Push, "Expected Push transition")
        
        # Convert expected states to names if they're classes
        expected_names = []
        for state in expected_states:
            if isinstance(state, type) and issubclass(state, State):
                expected_names.append(state.name())
            else:
                expected_names.append(str(state))
        
        actual_names = list(result.states)
        self.assertEqual(actual_names, expected_names,
                        f"Expected push states {expected_names}, got {actual_names}")
    
    def assertPopTransition(self, state_class, context=None, **context_kwargs):
        """Assert that a state returns a Pop transition."""
        result = run_state_on_state(state_class, context, **context_kwargs)
        self.assertIs(result, Pop, "Expected Pop transition")
    
    def assertStateRaisesException(self, state_class, expected_exception, 
                                  context=None, **context_kwargs):
        """Assert that a state raises a specific exception."""
        with self.assertRaises(expected_exception):
            run_state_on_state(state_class, context, **context_kwargs)
    
    def assertLifecycleMethodCalled(self, state_class, method_name, 
                                   context=None, **context_kwargs):
        """Assert that a specific lifecycle method can be called without error."""
        if context is None:
            context = make_mock_context(**context_kwargs)
        
        try:
            method = getattr(state_class, method_name)
            if method_name == 'on_state':
                # on_state must return something
                result = method(context)
                self.assertIsNotNone(result, f"{method_name} should return a transition")
            else:
                # Other lifecycle methods can return None
                result = method(context)
                # Just verify it doesn't crash
        except AttributeError:
            self.fail(f"State {state_class.__name__} does not have method {method_name}")
        except Exception as e:
            self.fail(f"Method {method_name} raised unexpected exception: {e}")
    
    def assertSharedStateModified(self, state_class, attribute_name, expected_value,
                                 context=None, **context_kwargs):
        """Assert that a state modifies shared state as expected."""
        if context is None:
            context = make_mock_context(**context_kwargs)
        
        # Run the state method
        run_state_on_state(state_class, context)
        
        # Check if the attribute was set correctly
        actual_value = getattr(context.common, attribute_name, None)
        self.assertEqual(actual_value, expected_value,
                        f"Expected {attribute_name}={expected_value}, got {actual_value}")
    
    def assertTransitionMapping(self, state_class, transition_enum, expected_target):
        """Assert that a state's transition mapping is correct."""
        transitions = state_class.transitions()
        
        self.assertIn(transition_enum, transitions, 
                     f"State {state_class.__name__} missing transition for {transition_enum}")
        
        actual_target = transitions[transition_enum]
        
        if isinstance(expected_target, type) and issubclass(expected_target, State):
            expected_name = expected_target.name()
            self.assertEqual(actual_target, expected_name,
                           f"Expected transition to {expected_name}, got {actual_target}")
        else:
            self.assertEqual(actual_target, expected_target,
                           f"Expected {expected_target}, got {actual_target}")
    
    def assertStateConfiguration(self, state_class, **expected_config):
        """Assert that a state's configuration matches expected values."""
        config = state_class.CONFIG
        
        for attr_name, expected_value in expected_config.items():
            actual_value = getattr(config, attr_name)
            self.assertEqual(actual_value, expected_value,
                           f"Expected {attr_name}={expected_value}, got {actual_value}")
    
    def assertTimeoutBehavior(self, state_class, context=None, **context_kwargs):
        """Assert that a state properly handles timeout messages."""
        timeout_msg = TimeoutMessage(state_class.name(), 1, 5.0)
        
        if context is None:
            context = make_mock_context(msg=timeout_msg, **context_kwargs)
        else:
            context.msg = timeout_msg
        
        # Test on_state with timeout message
        result = state_class.on_state(context)
        self.assertIsNotNone(result, "State should handle timeout message")
        
        # Test on_timeout directly
        timeout_result = state_class.on_timeout(context)
        self.assertIsNotNone(timeout_result, "on_timeout should return a transition")


class StateMachineTestCase(EnokiTestCase):
    """Extended test case for testing full StateMachine behavior."""
    
    def assertFSMTransition(self, initial_state, expected_state, 
                           messages=None, error_state=None, **sm_kwargs):
        """Assert that an FSM transitions from initial to expected state."""
        sm = StateMachine(initial_state, error_state=error_state, **sm_kwargs)
        
        # Send messages if provided
        if messages:
            for msg in messages:
                sm.send_message(msg)
        
        # Run one tick
        try:
            sm.tick()
        except BlockedInUntimedState:
            # This might be expected for certain tests
            pass
        except Exception as e:
            # Log the error but continue with assertion
            print(f"FSM tick error: {e}")
        
        if isinstance(expected_state, type) and issubclass(expected_state, State):
            expected_name = expected_state.name()
            actual_name = sm.current_state.name()
            self.assertEqual(actual_name, expected_name,
                           f"Expected FSM to be in {expected_name}, "
                           f"but it's in {actual_name}")
        else:
            self.assertEqual(sm.current_state, expected_state)
    
    def assertFSMCompletes(self, initial_state, messages=None, **sm_kwargs):
        """Assert that an FSM reaches a terminal state."""
        sm = StateMachine(initial_state, **sm_kwargs)
        
        # Send messages if provided
        if messages:
            for msg in messages:
                sm.send_message(msg)
        
        # Run until terminal or max iterations
        max_iterations = 100
        for i in range(max_iterations):
            if sm.current_state.CONFIG.terminal:
                break
            try:
                sm.tick()
            except BlockedInUntimedState:
                # Can't proceed further
                break
            except Exception:
                # Error occurred
                break
        
        self.assertTrue(sm.current_state.CONFIG.terminal, 
                       "FSM should reach terminal state")
    
    def assertFSMBlocks(self, initial_state, messages=None, **sm_kwargs):
        """Assert that an FSM blocks in a non-dwelling, non-timeout state."""
        sm = StateMachine(initial_state, **sm_kwargs)
        
        # Send messages if provided
        if messages:
            for msg in messages:
                sm.send_message(msg)
        
        with self.assertRaises(BlockedInUntimedState):
            sm.tick(block=False)
    
    def assertFSMStackState(self, sm, expected_stack):
        """Assert that the FSM's state stack matches expected states."""
        actual_stack = [s.name() if hasattr(s, 'name') else str(s) for s in sm.state_stack]
        expected_names = [s.name() if hasattr(s, 'name') else str(s) for s in expected_stack]
        self.assertEqual(actual_stack, expected_names,
                        f"Expected stack {expected_names}, got {actual_stack}")
    
    def assertValidationError(self, initial_state, error_state=None, **sm_kwargs):
        """Assert that StateMachine construction raises ValidationError."""
        with self.assertRaises(ValidationError):
            StateMachine(initial_state, error_state=error_state, **sm_kwargs)
    
    def assertFSMAnalysis(self, sm, expected_states=None, expected_groups=None):
        """Assert that FSM analysis produces expected results."""
        if expected_states:
            discovered = sm.discovered_states
            expected_names = {s.name() if hasattr(s, 'name') else str(s) 
                            for s in expected_states}
            self.assertEqual(discovered, expected_names,
                           f"Expected states {expected_names}, got {discovered}")
        
        if expected_groups:
            groups = sm.state_groups
            self.assertEqual(set(groups.keys()), set(expected_groups),
                           f"Expected groups {expected_groups}, got {list(groups.keys())}")


# Test Utilities for Complex Scenarios
class MockGeventQueue:
    """Mock gevent queue for testing without actual gevent operations."""
    
    def __init__(self):
        self._items = []
    
    def put(self, item):
        self._items.append(item)
    
    def get(self, timeout=None):
        if not self._items:
            if timeout is None:
                raise Exception("Queue empty")
            else:
                raise Exception("Timeout")
        return self._items.pop(0)
    
    def get_nowait(self):
        if not self._items:
            raise Exception("Queue empty")
        return self._items.pop(0)
    
    def empty(self):
        return len(self._items) == 0


def create_test_state_machine(initial_state, **kwargs):
    """
    Create a StateMachine suitable for testing, with mocked gevent components if needed.
    
    Args:
        initial_state: Initial state class
        **kwargs: Additional arguments for StateMachine
    
    Returns:
        StateMachine configured for testing
    """
    # In real tests, we'll use the actual StateMachine
    # This helper is for any special test configuration needed
    return StateMachine(initial_state, **kwargs)


def run_fsm_scenario(initial_state, messages=None, max_iterations=10, block=True, timeout=1, **sm_kwargs):
    """
    Run a complete FSM scenario and return execution trace.
    
    Args:
        initial_state: Initial state class
        messages: List of messages to send
        max_iterations: Maximum iterations to run
        **sm_kwargs: StateMachine constructor arguments
    
    Returns:
        Dict with execution results: {
            'final_state': final_state_class,
            'states_visited': [state_names],
            'completed': bool,
            'error': exception_or_none
        }
    """
    sm = StateMachine(initial_state, **sm_kwargs)
    states_visited = [sm.current_state.name()]
    error = None
    completed = False
    
    try:
        # Send all messages
        if messages:
            for msg in messages:
                sm.send_message(msg)
        
        # Run the state machine
        for i in range(max_iterations):
            if sm.current_state.CONFIG.terminal:
                completed = True
                break
            
            try:
                sm.tick(block=block, timeout=timeout)
                current_name = sm.current_state.name()
                if current_name != states_visited[-1]:
                    states_visited.append(current_name)
            except BlockedInUntimedState:
                # Can't proceed further
                raise
            except Exception as e:
                raise
    
    except Exception as e:
        error = e
    
    return {
        'final_state': sm.current_state,
        'states_visited': states_visited,
        'completed': completed,
        'error': error,
        'stack_depth': len(sm.state_stack),
        'ctx': sm.context
    }