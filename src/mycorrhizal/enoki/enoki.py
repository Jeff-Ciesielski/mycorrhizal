#!/usr/bin/env python3
"""
Enoki - A finite state machine framework

Key Features:
- States are collections of static/class methods (no instantiation)
- Declarative transitions via enums for static analysis
- String-based state resolution to break circular imports
- Gevent-based timeouts and message handling
- Comprehensive validation at construction time
- Push/Pop context resolution for accurate analysis
"""

import importlib
import importlib.util
import sys
import os
import types
import inspect

from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, Optional, Union, Type, Callable
import importlib
import importlib.util, sys, os, types, inspect
from itertools import accumulate
from pathlib import Path

import time
import inspect
import os
from functools import cache

import gevent
from gevent import spawn, sleep, Greenlet
from gevent.queue import (
    Queue as GeventQueue,
    Empty,
    PriorityQueue as GeventPriorityQueue,
)


class TransitionName(Enum):
    """Base class for state-specific transition enums"""

    pass


class GlobalTransitions(TransitionName):
    """Common transitions available to all states"""

    UNHANDLED = "unhandled"
    AGAIN = "again"
    REPEAT = "repeat"
    RESTART = "restart"
    RETRY = "retry"
    POP = "pop"
    TIMEOUT = "timeout"


@dataclass
class StateConfiguration:
    """Configuration for a state - replaces class attributes"""

    timeout: Optional[float] = None
    retries: Optional[int] = None
    terminal: bool = False
    can_dwell: bool = False


@dataclass
class SharedContext:
    """Context passed to state methods"""

    send_message: Callable
    log: Callable
    common: Any
    msg: Optional[Any] = None


@dataclass(order=True)
class PrioritizedMessage:
    priority: int
    item: Any = field(compare=False)


@dataclass
class EnokiInternalMessage:
    pass


@dataclass
class TimeoutMessage(EnokiInternalMessage):
    """Internal message for timeout events"""

    state_name: str
    timeout_id: int
    timeout_duration: float


class ValidationError(Exception):
    """Raised when state machine validation fails"""

    pass


class BlockedInUntimedState(Exception):
    """Raised when a non-dwelling state blocks without a timeout"""

    pass


class StateMachineComplete(Exception):
    """Raised when the state machine has reached a terminal state"""

    pass


class PopFromEmptyStack(Exception):
    """Raised when we attempt to pop from an empty stack"""


@dataclass
class ValidationResult:
    """Results of state machine validation"""

    valid: bool
    errors: list[str]
    warnings: list[str]
    discovered_states: set[str]


class StateMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Methods that should automatically become classmethods
        auto_classmethod_names = {
            "transitions",
            "on_state",
            "on_enter",
            "on_leave",
            "on_fail",
            "on_timeout",
        }

        # Convert specified methods to classmethods if they aren't already
        for method_name in auto_classmethod_names:
            if method_name in namespace:
                method = namespace[method_name]
                # Only convert if it's not already a classmethod/staticmethod
                if not isinstance(method, (classmethod, staticmethod)):
                    # If it's an abstractmethod, we need to handle it specially
                    if (
                        hasattr(method, "__isabstractmethod__")
                        and method.__isabstractmethod__
                    ):
                        # Create classmethod first, then make it abstract
                        new_method = classmethod(
                            abstractmethod(
                                method.__func__
                                if hasattr(method, "__func__")
                                else method
                            )
                        )
                    else:
                        new_method = classmethod(method)
                    namespace[method_name] = new_method

        # Also auto-convert any private methods (starting with _)
        for key, value in list(namespace.items()):
            if (
                key.startswith("_")
                and callable(value)
                and not isinstance(value, (classmethod, staticmethod))
                and not key.startswith("__")
            ):
                namespace[key] = classmethod(value)

        return super().__new__(mcs, name, bases, namespace)


class State(ABC, metaclass=StateMeta):
    """Base class for states"""

    CONFIG = StateConfiguration()

    @abstractmethod
    def transitions(
        cls,
    ) -> Dict[TransitionName, Union[str, Type["State"], "Push", "Pop"]]:
        """Returns mapping of transition enums to their targets"""
        return dict()

    @abstractmethod
    def on_state(cls, ctx: SharedContext) -> TransitionName:
        """Main state logic"""
        pass

    def on_enter(cls, ctx: SharedContext) -> None:
        """Called when entering the state"""
        pass

    def on_leave(cls, ctx: SharedContext) -> None:
        """Called when leaving the state"""
        pass

    def on_fail(cls, ctx: SharedContext) -> TransitionName:
        """Called when retry limit is exceeded"""
        return GlobalTransitions.UNHANDLED

    @abstractmethod
    def on_timeout(cls, ctx: SharedContext) -> TransitionName:
        """Called when state times out"""
        pass

    @classmethod
    @cache
    def name(cls) -> str:
        """Returns a fully qualified name for the state, omitting <locals> and using filename for scripts."""
        import inspect, os

        module = cls.__module__
        if module == "__main__":
            filename = inspect.getfile(cls)
            module = os.path.splitext(os.path.basename(filename))[0]
        return f"{module}.{cls.__qualname__}"

    @classmethod
    @cache
    def group_name(cls) -> str:
        """Returns the group name for visualization purposes"""
        for base in cls.__mro__:
            if base != State and issubclass(base, State) and base != cls:
                return base.__name__
        return cls.__name__

    @classmethod
    def base_name(cls) -> str:
        """Returns just the class name without module path"""
        return cls.__qualname__.split(".")[-1]


class Push:
    """Represents pushing states onto the stack"""

    def __init__(self, *states: Union[str, Type[State]]):
        self.states = states


class Pop:
    """Represents popping from the stack"""

    pass


class StateRegistry:
    """Manages state resolution and validation"""

    def __init__(self):
        self._state_cache: Dict[str, Type[State]] = {}
        self._transition_cache: Dict[str, Dict] = {}
        self._resolved_modules: Dict[str, Type] = {}

    def resolve_state(
        self, state_ref: Union[str, Type[State]], validate_only: bool = False
    ) -> Optional[Type[State]]:
        """Resolves a state reference to an actual state class, supporting nested classes/functions and dynamic import. Tracks all resolved file paths and modules for future resolution attempts."""

        if isinstance(state_ref, type) and issubclass(state_ref, State):
            # Track resolved module and short name
            module_name = getattr(state_ref, "__module__", None)
            if module_name and (module_name not in self._resolved_modules):
                mod = sys.modules.get(module_name)
                self._resolved_modules[module_name] = mod

                # Store a short name so we can do local file lookup more easily without needin the whole enchilada
                if mod and hasattr(mod, "__file__"):
                    fp = Path(getattr(mod, "__file__"))

                    # Now, grab just the filename, without extension and save that as well
                    self._resolved_modules[fp.stem] = mod

            return state_ref

        if isinstance(state_ref, str):
            if state_ref in self._state_cache:
                return self._state_cache[state_ref]

            try:
                if "." not in state_ref:
                    raise ValueError(
                        f"State reference '{state_ref}' must be fully qualified"
                    )

                module_paths = list(
                    accumulate(state_ref.split("."), lambda acc, x: acc + "." + x)
                )

                # See if the module exists in sys.modules
                mod_name = None
                for path in module_paths:
                    if path in self._resolved_modules:
                        mod_name = path
                        module = self._resolved_modules[mod_name]
                        break

                    if path in sys.modules:
                        mod_name = path
                        module = sys.modules[mod_name]
                        break

                if mod_name is None:
                    raise ValidationError(f"{state_ref} is nowhere in sys.modules")
                # Now, we have the module and its name, so the fq class name is
                # state_ref with the modue name clipped out
                fq_class_path = state_ref[len(mod_name) + 1 :].split(".")

                # Now, we need to walk down the fq_class_path, starting at the
                # module, getting attributes, and checking if either the
                # attribute itself exists, or if it exists within locals
                working_elt = module
                for path_elt in fq_class_path:
                    if hasattr(working_elt, path_elt):
                        working_elt = getattr(working_elt, path_elt)

                if not (
                    isinstance(working_elt, type) and issubclass(working_elt, State)
                ):
                    raise ValueError(f"'{state_ref}' is not a State subclass")

                if not validate_only:
                    self._state_cache[state_ref] = working_elt

                return working_elt

            except (ImportError, AttributeError, ValueError) as e:
                error_msg = f"Failed to resolve state '{state_ref}': {e}"
                if validate_only:
                    self._validation_errors.append(error_msg)
                    return None
                else:
                    raise ValidationError(error_msg)

        return state_ref

    @staticmethod
    def is_abstract_classmethod_implemented(cls, method_name):
        """Check if abstract method is implemented by checking __abstractmethods__"""
        return method_name not in cls.__abstractmethods__ and hasattr(cls, method_name)

    def validate_all_states(
        self,
        initial_state: Union[str, Type[State]],
        error_state: Optional[Union[str, Type[State]]] = None,
    ) -> ValidationResult:
        """Two-pass validation of all states in the state machine"""
        self._validation_errors = []
        self._validation_warnings = []

        # Pass 1: Discover all reachable states
        discovered_states = set()
        string_references = set()
        states_to_visit = []

        initial_resolved = self.resolve_state(initial_state, validate_only=True)
        if initial_resolved:
            states_to_visit.append(initial_resolved)
            discovered_states.add(initial_resolved.name())

        if error_state:
            error_resolved = self.resolve_state(error_state, validate_only=True)
            if error_resolved:
                states_to_visit.append(error_resolved)
                discovered_states.add(error_resolved.name())

        # Traverse all reachable states
        visited = set()
        while states_to_visit:
            current_state = states_to_visit.pop()
            state_name = current_state.name()

            if state_name in visited:
                continue
            visited.add(state_name)

            try:
                raw_transitions = current_state.transitions()

                for transition_enum, target in raw_transitions.items():
                    if not isinstance(transition_enum, TransitionName):
                        self._validation_errors.append(
                            f"State '{state_name}' has invalid transition key '{transition_enum}'"
                        )
                        continue

                    target_states = self._extract_state_references(target)
                    for target_state in target_states:
                        if isinstance(target_state, str):
                            string_references.add(target_state)
                        elif isinstance(target_state, type) and issubclass(
                            target_state, State
                        ):
                            target_name = target_state.name()
                            if target_name not in discovered_states:
                                discovered_states.add(target_name)
                                states_to_visit.append(target_state)

                # Validate that the state has an on_state handler
                if not self.is_abstract_classmethod_implemented(
                    current_state, "on_state"
                ):
                    self._validation_errors.append(
                        f"State {state_name} does not have an on_state handler defined"
                    )

                # Make sure non-terminal states define transitions
                if not current_state.CONFIG.terminal:
                    if not self.is_abstract_classmethod_implemented(
                        current_state, "transitions"
                    ):
                        self._validation_errors.append(
                            f"Non-terminal state {state_name} does not define any transitions"
                        )

                # Validate timeout configuration
                if current_state.CONFIG.timeout is not None:
                    if (
                        not isinstance(current_state.CONFIG.timeout, (int, float))
                        or current_state.CONFIG.timeout <= 0
                    ):
                        self._validation_errors.append(
                            f"State '{state_name}' has invalid timeout: {current_state.CONFIG.timeout}"
                        )

                    has_timeout_handler = self.is_abstract_classmethod_implemented(
                        current_state, "on_timeout"
                    )
                    if not has_timeout_handler:
                        self._validation_warnings.append(
                            f"State '{state_name}' has timeout but no on_timeout handler defined"
                        )

            except Exception as e:
                self._validation_errors.append(
                    f"Error processing transitions for state '{state_name}': {e}"
                )

        # Pass 2: Validate all string references
        for state_ref in string_references:
            resolved = self.resolve_state(state_ref, validate_only=True)
            if resolved:
                resolved_name = resolved.name()
                if resolved_name not in discovered_states:
                    discovered_states.add(resolved_name)
                    states_to_visit.append(resolved)

        # Validate newly discovered states
        while states_to_visit:
            current_state = states_to_visit.pop()
            state_name = current_state.name()

            if state_name in visited:
                continue
            visited.add(state_name)

            try:
                raw_transitions = current_state.transitions()
                for transition_enum, target in raw_transitions.items():
                    target_states = self._extract_state_references(target)
                    for target_state in target_states:
                        if isinstance(target_state, str):
                            if target_state not in string_references:
                                string_references.add(target_state)
                                resolved = self.resolve_state(
                                    target_state, validate_only=True
                                )
                                if resolved:
                                    resolved_name = resolved.name()
                                    if resolved_name not in discovered_states:
                                        discovered_states.add(resolved_name)
                                        states_to_visit.append(resolved)
                        elif isinstance(target_state, type) and issubclass(
                            target_state, State
                        ):
                            target_name = target_state.name()
                            if target_name not in discovered_states:
                                discovered_states.add(target_name)
                                states_to_visit.append(target_state)
            except Exception as e:
                self._validation_errors.append(
                    f"Error processing transitions for state '{state_name}': {e}"
                )

        return ValidationResult(
            valid=len(self._validation_errors) == 0,
            errors=self._validation_errors.copy(),
            warnings=self._validation_warnings.copy(),
            discovered_states=discovered_states,
        )

    def _extract_state_references(self, target: Any) -> list[Union[str, Type[State]]]:
        """Extract all state references from a transition target"""
        if isinstance(target, str):
            return [target]
        elif isinstance(target, type) and issubclass(target, State):
            return [target]
        elif isinstance(target, Push):
            return list(target.states)
        elif isinstance(target, GlobalTransitions):
            return []
        else:
            return []

    def get_transitions(self, state_class: Type[State]) -> Dict:
        """Gets the transition mapping for a state, with caching"""
        state_name = state_class.name()

        if state_name in self._transition_cache:
            return self._transition_cache[state_name]

        raw_transitions = state_class.transitions()
        resolved_transitions = {}

        for transition_enum, target in raw_transitions.items():
            if isinstance(target, str):
                resolved_transitions[transition_enum] = self.resolve_state(target)
            else:
                resolved_transitions[transition_enum] = target

        self._transition_cache[state_name] = resolved_transitions
        return resolved_transitions


class StateMachine:
    """State Machine with gevent-based async support"""

    class MessagePriorities(IntEnum):
        ERROR = 0
        INTERNAL_MESSAGE = 1
        MESSAGE = 2

    def __init__(
        self,
        initial_state: Union[str, Type[State]],
        error_state: Union[str, Type[State]] = None,
        filter_fn: Optional[callable] = None,
        trap_fn: Optional[callable] = None,
        on_error_fn=None,
        common_data: Optional[Any] = None,
    ):

        self.registry = StateRegistry()

        # The filter and trap functions are used to filter messages
        # (for example, common messages that apply to the process
        # rather than an individual state) and trap unhandled messages
        # (so that one could, for example, raise an exception)
        self._filter_fn = filter_fn or (lambda x, y: None)
        self._trap_fn = trap_fn or (lambda x: None)

        # The on_error function allows users to catch and handle specific kinds
        # of errors as they see fit and avoid crashing if the FSM can recover
        self._on_err_fn = on_error_fn or (lambda x, y: None)

        # Validate the state machine structure
        validation_result = self.registry.validate_all_states(
            initial_state, error_state
        )

        if not validation_result.valid:
            error_msg = "State machine validation failed:\n" + "\n".join(
                validation_result.errors
            )
            if validation_result.warnings:
                error_msg += "\n\nWarnings:\n" + "\n".join(validation_result.warnings)
            raise ValidationError(error_msg)

        # Log warnings
        if validation_result.warnings:
            for warning in validation_result.warnings:
                self.log(f"WARNING: {warning}")

        # Resolve initial states
        self.initial_state = self.registry.resolve_state(initial_state)
        self.error_state = (
            self.registry.resolve_state(error_state) if error_state else None
        )

        # Pre-compute analysis
        self._state_analysis = self._analyze_all_states()

        # Gevent-based infrastructure
        self._message_queue = GeventPriorityQueue()
        self._timeout_greenlet = None
        self._timeout_counter = 0
        self._current_timeout_id = None

        # Runtime state
        self.current_state = None
        self.state_stack = []
        self.context = SharedContext(
            send_message=self.send_message,
            log=self.log,
            common=common_data or dict(),
        )

        self.retry_counters = {}
        self.state_enter_times = {}

        self.reset()

    def _analyze_all_states(self) -> Dict[str, Dict]:
        """Pre-compute analysis of all states using graph traversal with stack simulation"""
        analysis = {}

        # Phase 1: Discover all states and their raw transitions
        visited = set()
        to_visit = [self.initial_state]

        if self.error_state:
            to_visit.append(self.error_state)

        # First pass: collect all states and their raw transition data
        while to_visit:
            current = to_visit.pop()
            state_name = current.name()

            if state_name in visited:
                continue

            visited.add(state_name)
            transitions = self.registry.get_transitions(current)

            # Store raw transition data for later processing
            raw_transitions = {}
            for transition_enum, target in transitions.items():
                raw_transitions[transition_enum] = target

            analysis[state_name] = {
                "state_class": current,
                "raw_transitions": raw_transitions,
                "transitions": {},  # Will be filled in phase 3
                "targets": [],  # Will be filled in phase 3
                "possible_pop_targets": set(),  # All states this could pop to
                "push_sources": [],  # All push operations that can reach this state
                "config": {
                    "timeout": current.CONFIG.timeout,
                    "retries": current.CONFIG.retries,
                    "terminal": current.CONFIG.terminal,
                    "can_dwell": current.CONFIG.can_dwell,
                },
                "group_name": current.group_name(),
                "base_name": current.base_name(),
            }

            # Add all referenced states to visit queue
            for transition_enum, target in transitions.items():
                if isinstance(target, type) and issubclass(target, State):
                    if target not in to_visit and target.name() not in visited:
                        to_visit.append(target)
                elif isinstance(target, Push):
                    for state_ref in target.states:
                        resolved = self.registry.resolve_state(state_ref)
                        if (
                            resolved
                            and resolved not in to_visit
                            and resolved.name() not in visited
                        ):
                            to_visit.append(resolved)

        # Phase 2: Graph traversal with stack simulation
        # We'll do a BFS/DFS to find all possible paths and their stack states

        class PathState:
            """Represents a state in our graph traversal"""

            def __init__(self, state_name: str, stack: list, path: list):
                self.state_name = state_name
                self.stack = stack.copy()  # Current stack contents
                self.path = path.copy()  # Path taken to get here

        # Track all the ways we can reach each state
        state_contexts = {}  # state_name -> list of (stack, path) tuples

        # Start traversal from initial state
        queue = [PathState(self.initial_state.name(), [], [])]
        if self.error_state:
            queue.append(PathState(self.error_state.name(), [], []))

        # Limit traversal to prevent infinite loops
        max_path_length = 50
        seen_states = set()  # (state_name, tuple(stack)) to detect cycles

        while queue:
            current = queue.pop(0)

            # Skip if path is too long (cycle detection)
            if len(current.path) > max_path_length:
                continue

            # Create a hashable representation of this state+stack combination
            stack_tuple = tuple(current.stack)
            state_key = (current.state_name, stack_tuple)

            # Skip if we've seen this exact state+stack combination
            if state_key in seen_states:
                continue
            seen_states.add(state_key)

            # Record this context for the state
            if current.state_name not in state_contexts:
                state_contexts[current.state_name] = []
            state_contexts[current.state_name].append(
                {"stack": current.stack, "path": current.path}
            )

            # Get transitions for current state
            state_info = analysis.get(current.state_name)
            if not state_info:
                continue

            # Process each transition
            for transition_enum, target in state_info["raw_transitions"].items():
                transition_name = (
                    transition_enum.name
                    if hasattr(transition_enum, "name")
                    else str(transition_enum)
                )

                if isinstance(target, type) and issubclass(target, State):
                    # Simple transition - same stack
                    next_path = current.path + [(current.state_name, transition_name)]
                    queue.append(PathState(target.name(), current.stack, next_path))

                elif isinstance(target, Push):
                    # Push transition - modify stack
                    resolved_states = []
                    for state_ref in target.states:
                        resolved = self.registry.resolve_state(state_ref)
                        if resolved:
                            resolved_states.append(resolved.name())

                    if resolved_states:
                        # First state is where we transition to
                        next_state = resolved_states[0]
                        # Rest go on the stack
                        new_stack = resolved_states[1:] + current.stack
                        next_path = current.path + [
                            (current.state_name, transition_name)
                        ]
                        queue.append(PathState(next_state, new_stack, next_path))

                        # Record that this push can reach all states in the push
                        for i, pushed_state in enumerate(resolved_states):
                            if pushed_state not in analysis:
                                continue
                            analysis[pushed_state]["push_sources"].append(
                                {
                                    "from_state": current.state_name,
                                    "transition": transition_name,
                                    "push_states": resolved_states,
                                    "position": i,
                                }
                            )

                elif target is Pop:
                    # Pop transition - use stack
                    if current.stack:
                        next_state = current.stack[0]
                        new_stack = current.stack[1:]
                        next_path = current.path + [
                            (current.state_name, transition_name)
                        ]
                        queue.append(PathState(next_state, new_stack, next_path))

                        # Record this as a possible pop target
                        analysis[current.state_name]["possible_pop_targets"].add(
                            next_state
                        )
                    # If stack is empty, this path ends here (but don't mark as error yet)

        # Phase 3: Build final transition mappings with all pop targets
        validation_issues = []

        for state_name, state_info in analysis.items():
            targets = []
            transition_details = {}

            # Get all contexts where this state appears
            contexts = state_contexts.get(state_name, [])

            for transition_enum, target in state_info["raw_transitions"].items():
                transition_name = (
                    transition_enum.name
                    if hasattr(transition_enum, "name")
                    else str(transition_enum)
                )

                if isinstance(target, type) and issubclass(target, State):
                    target_name = target.name()
                    targets.append(target_name)
                    transition_details[transition_name] = target_name

                elif isinstance(target, Push):
                    # Resolve all states in the push
                    resolved_states = []
                    for state_ref in target.states:
                        resolved = self.registry.resolve_state(state_ref)
                        if resolved:
                            resolved_states.append(resolved.name())

                    if resolved_states:
                        # First state is the immediate target
                        targets.append(resolved_states[0])
                        # All states are reachable
                        targets.extend(resolved_states[1:])

                        transition_details[transition_name] = (
                            f"Push({', '.join(resolved_states)})"
                        )

                elif target is Pop:
                    # Get all possible pop targets from our analysis
                    pop_targets = state_info["possible_pop_targets"]

                    if not pop_targets:
                        # No valid pop targets found in any context
                        if not contexts:
                            transition_details[transition_name] = (
                                "Pop -> <unreachable state>"
                            )
                            validation_issues.append(
                                f"State '{state_name}' is unreachable but has Pop transition"
                            )
                        else:
                            # State is reachable but all paths have empty stack
                            transition_details[transition_name] = (
                                "Pop -> <stack underflow>"
                            )
                            validation_issues.append(
                                f"State '{state_name}' always has empty stack at Pop"
                            )
                    elif len(pop_targets) == 1:
                        # Single pop target
                        target_name = list(pop_targets)[0]
                        targets.append(target_name)
                        transition_details[transition_name] = f"Pop -> {target_name}"
                    else:
                        # Multiple pop targets - this is valid!
                        # The actual target depends on runtime stack state
                        targets.extend(pop_targets)
                        target_list = sorted(pop_targets)
                        transition_details[transition_name] = (
                            f"Pop -> {{{', '.join(target_list)}}}"
                        )

                elif isinstance(target, GlobalTransitions):
                    transition_details[transition_name] = f"Global.{target.name}"

                else:
                    transition_details[transition_name] = str(target)

            state_info["transitions"] = transition_details
            state_info["targets"] = list(set(targets))  # Remove duplicates
            state_info["contexts"] = contexts  # Store all contexts for analysis

        # Phase 4: Validation - check for issues
        # Check for infinite push cycles
        for state_name, contexts in state_contexts.items():
            max_stack_depth = (
                max(len(ctx["stack"]) for ctx in contexts) if contexts else 0
            )
            if max_stack_depth > 20:
                validation_issues.append(
                    f"State '{state_name}' can have very deep stack (max: {max_stack_depth}) - possible infinite push cycle"
                )

        # Report validation issues
        if validation_issues:
            self.log("Push/Pop Analysis Issues:")
            for issue in validation_issues:
                self.log(f"  - {issue}")

        return analysis

    def send_message(self, message: Any):
        """Send a message to the state machine"""
        try:
            if message and self._filter_fn(self.context, message):
                return
            self._send_message_internal(message)
        except Exception as e:
            self._send_message_internal(e)

    def _send_message_internal(self, message):
        match message:
            case _ if isinstance(message, EnokiInternalMessage):
                self._message_queue.put(
                    PrioritizedMessage(self.MessagePriorities.INTERNAL_MESSAGE, message)
                )
            case _ if isinstance(message, Exception):
                self._message_queue.put(
                    PrioritizedMessage(self.MessagePriorities.ERROR, message)
                )
            case _:
                # standard message
                self._message_queue.put(
                    PrioritizedMessage(self.MessagePriorities.MESSAGE, message)
                )

    def _send_timeout_message(self, state_name: str, timeout_id: int, duration: float):
        """Internal method to send timeout messages"""
        timeout_msg = TimeoutMessage(state_name, timeout_id, duration)
        self._send_message_internal(timeout_msg)

    def _start_timeout(self, state_class: Type[State]) -> Optional[int]:
        """Start a timeout for the given state"""
        if state_class.CONFIG.timeout is None:
            return None

        self._cancel_timeout()

        self._timeout_counter += 1
        timeout_id = self._timeout_counter
        self._current_timeout_id = timeout_id

        self._timeout_greenlet = spawn(
            self._timeout_handler,
            state_class.name(),
            timeout_id,
            state_class.CONFIG.timeout,
        )

        return timeout_id

    def _timeout_handler(self, state_name: str, timeout_id: int, duration: float):
        """Gevent greenlet that handles timeout"""
        try:
            sleep(duration)
            self._send_timeout_message(state_name, timeout_id, duration)
        except Exception:
            pass  # Greenlet was killed (timeout cancelled)

    def _cancel_timeout(self):
        """Cancel any active timeout"""
        if self._timeout_greenlet is not None:
            self._timeout_greenlet.kill()
            self._timeout_greenlet = None
        self._current_timeout_id = None

    def _handle_timeout_message(self) -> bool:
        """Handle a timeout message"""
        timeout_msg = self.context.msg
        self.context.msg = None

        if timeout_msg.timeout_id != self._current_timeout_id:
            self.log(f"Ignoring stale timeout {timeout_msg.timeout_id}")
            return False

        if timeout_msg.state_name != self.current_state.name():
            self.log(
                f"Ignoring timeout for {timeout_msg.state_name}, current state is {self.current_state.name()}"
            )
            return False

        self.log(
            f"Handling timeout for {timeout_msg.state_name} ({timeout_msg.timeout_duration}s)"
        )
        self._cancel_timeout()
        return True

    def reset(self):
        """Reset the state machine to initial state"""
        self._cancel_timeout()
        self.state_stack = []
        self.retry_counters = {}
        self.state_enter_times = {}

        # Clear message queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except:
                break

        # Transition into our initial state
        self._transition_to_state(self.initial_state)

    def log(self, message: str):
        """Log a message"""
        print(f"[FSM] {message}")

    def tick(self, timeout: Optional[float] = 0):
        """Process one state machine tick"""
        if not self.current_state:
            return

        match timeout:
            case 0:
                block = False
            case None:
                block = True
            case x if x > 0:
                block = True
            case _:
                raise ValueError("Tick timeout must be None, or >=0")

        while True:
            try:
                if isinstance(self.context.msg, Exception):
                    raise self.context.msg

                # Handle timeout messages specially
                if isinstance(self.context.msg, TimeoutMessage):
                    if not self._handle_timeout_message():
                        return
                    transition = self.current_state.on_timeout(self.context)
                else:
                    transition = self.current_state.on_state(self.context)
                if self.current_state.CONFIG.terminal:
                    raise StateMachineComplete

                if transition in (None, GlobalTransitions.UNHANDLED):
                    self._trap_fn(self.context)

                self.context.msg = None

                should_check_queue = self._process_transition(transition)

                if should_check_queue:
                    if block:
                        message = self._message_queue.get(timeout=timeout).item
                    else:
                        message = self._message_queue.get_nowait().item
                    self.context.msg = message
            except Empty:
                break
            except Exception as e:
                # While it's true that 'Pokemon errors' are typically
                # in poor taste, this allows the user to selectively
                # handle error cases, and throw any error that isn't
                # explicitely handled

                # If we're terminal anyway, bail
                if (
                    isinstance(e, StateMachineComplete)
                    or self.current_state.CONFIG.terminal
                ):
                    raise

                next_transition = self._on_err_fn(self.context, e)
                if next_transition:
                    self._process_transition(next_transition)
                else:
                    raise

    def _process_transition(self, transition: TransitionName) -> bool:
        """Process a state transition"""
        transitions = self.registry.get_transitions(self.current_state)

        if transition and (transition not in transitions):
            self.log(f"ERROR: Unknown transition {transition}")
            return

        target = transitions.get(transition)

        if transition in (None, GlobalTransitions.UNHANDLED):
            if (
                self.current_state.CONFIG.can_dwell
            ) or self.current_state.CONFIG.timeout:
                return True
            raise BlockedInUntimedState(
                f"{self.current_state.name} cannot dwell and does not have a timeout"
            )
        elif isinstance(target, type) and issubclass(target, State):
            self._transition_to_state(target)
        elif isinstance(target, Push):
            self._handle_push(target)
        elif isinstance(target, Pop) or issubclass(target, Pop):
            self._handle_pop()
        elif target == GlobalTransitions.AGAIN:
            spawn(self.tick, block=False)
        elif target == GlobalTransitions.REPEAT:
            self._transition_to_state(type(self.current_state))
        elif target == GlobalTransitions.RESTART:
            self._reset_state_context()
            self._transition_to_state(type(self.current_state))
            return True
        elif target == GlobalTransitions.RETRY:
            self._handle_retry()

        return False

    def _transition_to_state(self, next_state: Type[State]):
        """Transition to a new state"""
        if self.current_state:
            self.current_state.on_leave(self.context)
            self._cancel_timeout()

        self.current_state = next_state
        self.state_enter_times[next_state.name()] = time.time()
        self.log(f"Transitioned to {next_state.name()}")

        self._start_timeout(next_state)
        next_state.on_enter(self.context)

    def _handle_retry(self):
        """Handle retry logic with counter"""
        state_name = self.current_state.name()

        if state_name not in self.retry_counters:
            self.retry_counters[state_name] = 0

        self.retry_counters[state_name] += 1

        if (
            self.current_state.CONFIG.retries is not None
            and self.retry_counters[state_name] > self.current_state.CONFIG.retries
        ):
            self.log(f"Retry limit exceeded for {state_name}")
            transition = self.current_state.on_fail(self.context)
            self._process_transition(transition)
        else:
            self._transition_to_state(type(self.current_state))

    def _reset_state_context(self):
        """Reset context for current state"""
        state_name = self.current_state.name()
        if state_name in self.retry_counters:
            del self.retry_counters[state_name]

    def _handle_push(self, push: Push):
        """Handle push transition"""
        resolved_states = [self.registry.resolve_state(s) for s in push.states]

        for state in reversed(resolved_states[1:]):
            self.state_stack.append(state)

        self._transition_to_state(resolved_states[0])

    def _handle_pop(self):
        """Handle pop transition"""
        if not self.state_stack:
            raise PopFromEmptyStack

        next_state = self.state_stack.pop()
        self._transition_to_state(next_state)

    def run(self, max_iterations: Optional[int] = None, timeout: Optional[float] = 1):
        """Run the state machine until terminal state"""
        iteration = 0

        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            if self.current_state.CONFIG.terminal:
                self.log(f"Reached terminal state: {self.current_state.name()}")
                break

            try:
                self.tick(timeout=timeout)
                iteration += 1
            except StateMachineComplete:
                raise
            except Exception as e:
                self.log(f"Error in FSM: {e}")
                if self.error_state:
                    self._transition_to_state(self.error_state)
                else:
                    raise

    @property
    def discovered_states(self) -> set[str]:
        """Get all states discovered during analysis"""
        return set(self._state_analysis.keys())

    @property
    def state_groups(self) -> Dict[str, list[str]]:
        """Get states organized by their group names"""
        groups = {}
        for state_name, info in self._state_analysis.items():
            group_name = info["group_name"]
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(state_name)
        return groups

    def get_state_info(self, state_name: str) -> Dict:
        """Get analysis information for a specific state"""
        return self._state_analysis.get(state_name, {})

    def get_push_pop_relationships(self) -> Dict[str, Dict]:
        """Get detailed push/pop relationships from the analysis"""
        relationships = {}

        for state_name, info in self._state_analysis.items():
            # Get all possible pop targets
            pop_targets = []
            for transition_name, target_str in info["transitions"].items():
                if "Pop ->" in target_str:
                    if "{" in target_str:
                        # Multiple targets: "Pop -> {A, B, C}"
                        targets_str = target_str[
                            target_str.index("{") + 1 : target_str.rindex("}")
                        ]
                        targets = [t.strip() for t in targets_str.split(",")]
                        pop_targets.append(
                            {
                                "transition": transition_name,
                                "targets": targets,
                                "is_multi": True,
                            }
                        )
                    elif "<" not in target_str:
                        # Single target: "Pop -> A"
                        target = target_str.split(" -> ")[1].strip()
                        pop_targets.append(
                            {
                                "transition": transition_name,
                                "targets": [target],
                                "is_multi": False,
                            }
                        )
                    else:
                        # Error case: "Pop -> <error>"
                        pop_targets.append(
                            {
                                "transition": transition_name,
                                "targets": [],
                                "is_multi": False,
                                "error": target_str.split(" -> ")[1].strip(),
                            }
                        )

            # Get contexts
            contexts = info.get("contexts", [])
            unique_stacks = set()
            for ctx in contexts:
                unique_stacks.add(tuple(ctx["stack"]))

            relationships[state_name] = {
                "push_sources": info.get("push_sources", []),
                "pop_targets": pop_targets,
                "possible_pop_destinations": sorted(
                    info.get("possible_pop_targets", set())
                ),
                "is_pushed_to": len(info.get("push_sources", [])) > 0,
                "has_pop": len(pop_targets) > 0,
                "has_multi_pop": any(pt.get("is_multi", False) for pt in pop_targets),
                "reachable_contexts": len(contexts),
                "unique_stack_states": len(unique_stacks),
                "max_stack_depth": (
                    max(len(ctx["stack"]) for ctx in contexts) if contexts else 0
                ),
            }

        return relationships

    def get_reachable_states(
        self, from_state: Union[str, Type[State]], max_depth: Optional[int] = None
    ) -> set[str]:
        """Get all states reachable from a given starting state"""
        if isinstance(from_state, type):
            start_name = from_state.name()
        else:
            start_name = from_state

        if start_name not in self._state_analysis:
            return set()

        reachable = set()
        to_visit = [(start_name, 0)]

        while to_visit:
            current_name, depth = to_visit.pop()

            if current_name in reachable:
                continue

            if max_depth is not None and depth > max_depth:
                continue

            reachable.add(current_name)

            current_info = self._state_analysis[current_name]
            for target_name in current_info["targets"]:
                if target_name not in reachable:
                    to_visit.append((target_name, depth + 1))

        return reachable

    def print_push_pop_analysis(
        self, state_filter: Optional[Union[str, Type[State]]] = None
    ):
        """
        Print a detailed push/pop analysis for all states or a specific state.

        Args:
            state_filter: Optional state name or class to filter the analysis.
                        If None, prints analysis for all states with push/pop relationships.
        """
        relationships = self.get_push_pop_relationships()

        # Resolve state filter if it's a class
        if (
            state_filter
            and isinstance(state_filter, type)
            and issubclass(state_filter, State)
        ):
            state_filter = state_filter.name()

        # Determine which states to analyze
        states_to_analyze = []
        if state_filter:
            if state_filter in relationships:
                states_to_analyze = [state_filter]
            else:
                print(f"State '{state_filter}' not found in analysis")
                return
        else:
            # Include all states that have push/pop relationships
            states_to_analyze = [
                state_name
                for state_name, rel_info in relationships.items()
                if rel_info["is_pushed_to"] or rel_info["has_pop"]
            ]

        if not states_to_analyze:
            print("No states with push/pop relationships found")
            return

        print("\nPush/Pop Analysis:")
        print("=" * 60)

        for state_name in sorted(states_to_analyze):
            rel_info = relationships[state_name]
            state_info = self._state_analysis.get(state_name, {})

            print(f"\n{state_name}:")

            # Show what pushes to this state
            if rel_info["push_sources"]:
                print("  Pushed by:")
                # Group by source state for cleaner output
                sources_by_state = {}
                for source in rel_info["push_sources"]:
                    key = source["from_state"]
                    if key not in sources_by_state:
                        sources_by_state[key] = []
                    sources_by_state[key].append(source)

                for source_state, pushes in sorted(sources_by_state.items()):
                    for push in pushes:
                        print(f"    - {source_state} via {push['transition']}")
                        print(f"      Push group: {push['push_states']}")
                        print(f"      Position in group: {push['position']}")

            # Show pop targets
            if rel_info["pop_targets"]:
                print("  Pop transitions:")
                for pop in rel_info["pop_targets"]:
                    if pop.get("error"):
                        print(f"    - {pop['transition']}: {pop['error']}")
                    elif pop.get("is_multi"):
                        print(
                            f"    - {pop['transition']}: Can pop to any of {pop['targets']}"
                        )
                    else:
                        print(f"    - {pop['transition']}: Pops to {pop['targets'][0]}")

            # Show stack context information
            if rel_info["reachable_contexts"] > 0:
                print(f"  Context information:")
                print(
                    f"    - Reachable via {rel_info['reachable_contexts']} different paths"
                )
                print(
                    f"    - {rel_info['unique_stack_states']} unique stack configurations"
                )
                print(f"    - Maximum stack depth: {rel_info['max_stack_depth']}")

                # Show possible stacks (first few examples)
                contexts = state_info.get("contexts", [])
                if contexts and rel_info["unique_stack_states"] > 0:
                    print(f"    - Example stack states:")
                    shown = set()
                    count = 0
                    for ctx in contexts:
                        stack_tuple = tuple(ctx["stack"])
                        if stack_tuple not in shown and count < 3:
                            shown.add(stack_tuple)
                            if ctx["stack"]:
                                print(f"      • {ctx['stack']}")
                            else:
                                print(f"      • [empty]")
                            count += 1
                    if rel_info["unique_stack_states"] > 3:
                        print(
                            f"      • ... and {rel_info['unique_stack_states'] - 3} more"
                        )

            # Show summary
            if rel_info["is_pushed_to"] or rel_info["has_pop"]:
                summary_parts = []
                if rel_info["is_pushed_to"]:
                    summary_parts.append(
                        f"pushed to by {len(rel_info['push_sources'])} transition(s)"
                    )
                if rel_info["has_pop"]:
                    if rel_info["has_multi_pop"]:
                        summary_parts.append("has context-dependent pop")
                    else:
                        summary_parts.append("has pop")
                print(f"  Summary: {', '.join(summary_parts)}")

    def print_state_stack_trace(self, state_name: Union[str, Type[State]]):
        """
        Print example execution traces showing how to reach a state and what happens with Pop.

        Args:
            state_name: State to analyze
        """
        # Resolve state name if it's a class
        if isinstance(state_name, type) and issubclass(state_name, State):
            state_name = state_name.name()

        state_info = self._state_analysis.get(state_name)
        if not state_info:
            print(f"State '{state_name}' not found")
            return

        contexts = state_info.get("contexts", [])
        if not contexts:
            print(f"State '{state_name}' is not reachable from initial state")
            return

        print(f"\nExecution traces for {state_name}:")
        print("=" * 60)

        # Show up to 3 different paths
        shown_paths = set()
        examples_shown = 0

        for ctx in contexts:
            if examples_shown >= 3:
                break

            # Create a simplified path representation to avoid duplicates
            path_key = tuple(
                (step[0], step[1]) for step in ctx["path"][-5:]
            )  # Last 5 steps
            if path_key in shown_paths:
                continue
            shown_paths.add(path_key)
            examples_shown += 1

            print(f"\nExample {examples_shown}:")
            print(f"  Path to reach {state_name}:")

            # Show last few steps of the path
            path_to_show = ctx["path"][-5:] if len(ctx["path"]) > 5 else ctx["path"]
            if len(ctx["path"]) > 5:
                print(f"    ... ({len(ctx['path']) - 5} earlier steps)")

            for i, (from_state, transition) in enumerate(path_to_show):
                print(f"    {i+1}. {from_state} --[{transition}]--> ", end="")
                if i < len(path_to_show) - 1:
                    print(path_to_show[i + 1][0])
                else:
                    print(state_name)

            print(
                f"  Stack when entering {state_name}: {ctx['stack'] if ctx['stack'] else '[empty]'}"
            )

            # Show what happens with Pop transitions
            rel_info = self.get_push_pop_relationships().get(state_name, {})
            if rel_info["has_pop"] and ctx["stack"]:
                print(f"  If Pop is triggered:")
                print(f"    → Would transition to: {ctx['stack'][0]}")
                print(
                    f"    → New stack would be: {ctx['stack'][1:] if len(ctx['stack']) > 1 else '[empty]'}"
                )

        if len(contexts) > 3:
            print(f"\n... and {len(contexts) - 3} more possible execution paths")

    def generate_mermaid_flowchart(
        self,
        from_state: Optional[Union[str, Type[State]]] = None,
        max_depth: Optional[int] = None,
        include_groups: bool = True,
        show_all_pop_targets: bool = True,
    ) -> str:
        """Generate a Mermaid flowchart showing all transitions including multiple pop targets"""
        if from_state is not None:
            states_to_include = self.get_reachable_states(from_state, max_depth)
        else:
            states_to_include = set(self._state_analysis.keys())

        filtered_analysis = {
            name: info
            for name, info in self._state_analysis.items()
            if name in states_to_include
        }

        result = "flowchart TD\n"

        # Generate nodes
        if include_groups:
            groups = {}
            for state_name, info in filtered_analysis.items():
                group_name = info["group_name"]
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append((state_name, info))

            for group_name, states in groups.items():
                if len(states) > 1:
                    result += f'\n    subgraph {group_name}Group["{group_name}"]\n'

                    for state_name, info in states:
                        clean_name = state_name.replace(".", "_")
                        base_name = info["base_name"]

                        timeout_info = ""
                        if info["config"]["timeout"]:
                            timeout_info = f" ({info['config']['timeout']}s)"

                        if info["config"]["terminal"]:
                            result += f'        {clean_name}["{base_name}{timeout_info}"]:::terminal\n'
                        else:
                            result += (
                                f'        {clean_name}["{base_name}{timeout_info}"]\n'
                            )

                    result += "    end\n"
                else:
                    state_name, info = states[0]
                    clean_name = state_name.replace(".", "_")
                    base_name = info["base_name"]

                    timeout_info = ""
                    if info["config"]["timeout"]:
                        timeout_info = f" ({info['config']['timeout']}s)"

                    if info["config"]["terminal"]:
                        result += f'    {clean_name}["{base_name}{timeout_info}"]:::terminal\n'
                    else:
                        result += f'    {clean_name}["{base_name}{timeout_info}"]\n'
        else:
            for state_name, info in filtered_analysis.items():
                clean_name = state_name.replace(".", "_")
                base_name = info["base_name"]

                timeout_info = ""
                if info["config"]["timeout"]:
                    timeout_info = f" ({info['config']['timeout']}s)"

                if info["config"]["terminal"]:
                    result += (
                        f'    {clean_name}["{base_name}{timeout_info}"]:::terminal\n'
                    )
                else:
                    result += f'    {clean_name}["{base_name}{timeout_info}"]\n'

        # Add transitions
        result += "\n"
        for state_name, info in filtered_analysis.items():
            clean_name = state_name.replace(".", "_")

            for transition_name, target_str in info["transitions"].items():
                if target_str.startswith("Global."):
                    continue

                if target_str.startswith("Push("):
                    # Handle Push transitions
                    states_in_push = target_str[5:-1].split(", ")
                    if states_in_push and states_in_push[0] in states_to_include:
                        clean_target = states_in_push[0].replace(".", "_")
                        result += f'    {clean_name} -->|"{transition_name} (Push)"| {clean_target}\n'

                elif target_str.startswith("Pop -> {") and show_all_pop_targets:
                    # Multiple pop targets
                    targets_str = target_str[
                        target_str.index("{") + 1 : target_str.rindex("}")
                    ]
                    targets = [t.strip() for t in targets_str.split(",")]

                    for target in targets:
                        if target in states_to_include:
                            clean_target = target.replace(".", "_")
                            result += f'    {clean_name} -->|"{transition_name} (Pop)"| {clean_target}\n'

                elif target_str.startswith("Pop -> ") and not target_str[7:].startswith(
                    "<"
                ):
                    # Single pop target
                    pop_target = target_str[7:]
                    if pop_target in states_to_include:
                        clean_target = pop_target.replace(".", "_")
                        result += f'    {clean_name} -->|"{transition_name} (Pop)"| {clean_target}\n'

                elif target_str in states_to_include:
                    # Regular transition
                    clean_target = target_str.replace(".", "_")
                    result += (
                        f'    {clean_name} -->|"{transition_name}"| {clean_target}\n'
                    )

        result += "\n    classDef terminal fill:#ff6b6b\n"
        return result
