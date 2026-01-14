#!/usr/bin/env python3
"""
Enoki - Asyncio Finite State Machine Framework

A decorator-based DSL for defining and executing state machines with support for
asyncio, timeouts, message passing, and hierarchical state composition.

Usage:
    from mycorrhizal.enoki.core import enoki, StateMachine, LabeledTransition
    from enum import Enum, auto

    @enoki.state()
    def IdleState():
        class Events(Enum):
            START = auto()
            QUIT = auto()

        @enoki.on_state
        async def on_state(ctx):
            if ctx.msg == "start":
                return Events.START
            return None

        @enoki.transitions
        def transitions():
            return [
                LabeledTransition(Events.START, ProcessingState),
                LabeledTransition(Events.QUIT, DoneState),
            ]

    # Create and run the FSM
    fsm = StateMachine(initial_state=IdleState, common_data={})
    await fsm.initialize()
    fsm.send_message("start")
    await fsm.tick()

Key Classes:
    StateMachine - Main FSM runtime with message queue and tick-based execution
    StateConfiguration - Configuration for states (timeout, retries, terminal, can_dwell)
    SharedContext - Context object passed to state handlers with msg and common data
    LabeledTransition - Maps events to target states

Transition Types:
    State references - Direct transition to another state
    Again - Re-execute current state immediately
    Unhandled - Wait for next message
    Retry - Re-enter state with retry counter
    Restart - Reset retry counter and wait for message
    Repeat - Re-enter state from on_enter
    Push(state1, state2, ...) - Push states onto stack
    Pop - Pop and return to previous state

State Handlers:
    on_state(ctx) - Main state logic, return transition or None to wait
    on_enter(ctx) - Called when entering state
    on_leave(ctx) - Called when leaving state
    on_timeout(ctx) - Called if timeout expires
    on_fail(ctx) - Called if exception occurs
"""

from __future__ import annotations
import inspect
import sys
import os
import time
import traceback
import asyncio
from pathlib import Path
from asyncio import PriorityQueue
from dataclasses import dataclass, field
from enum import Enum, auto, IntEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Awaitable,
    TypeVar,
    Generic,
    Tuple,
    Type,
)
from functools import cache


T = TypeVar('T')


# ============================================================================
# Interface Integration Helper
# ============================================================================

# Cache for interface views to avoid repeated creation
_interface_view_cache: Dict[Tuple[int, Type], Any] = {}


def _clear_interface_view_cache() -> None:
    """Clear the interface view cache. Useful for testing."""
    global _interface_view_cache
    _interface_view_cache.clear()


def _create_interface_view_for_context(context: SharedContext, handler: Callable) -> SharedContext:
    """
    Create a constrained view of context.common if the handler has an interface type hint.

    This enables type-safe, constrained access to blackboard state based on
    interface definitions created with @blackboard_interface.

    The function signature can use an interface type:
        async def on_state(ctx: SharedContext[MyInterface]):
            # ctx.common is automatically a constrained view
            return Events.DONE

    Args:
        context: The SharedContext object
        handler: The state handler function to check for interface type hints

    Returns:
        Either the original context or a new context with constrained common field
    """
    from typing import get_type_hints

    try:
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())

        # Check first parameter (should be SharedContext)
        if params and params[0].name == 'ctx':
            ctx_type = get_type_hints(handler).get('ctx')

            # If type hint exists and is a generic SharedContext with interface
            if ctx_type and hasattr(ctx_type, '__args__'):
                # Extract the type argument from SharedContext[InterfaceType]
                interface_type = ctx_type.__args__[0]

                # If it has interface metadata, create constrained view
                if hasattr(interface_type, '_readonly_fields'):
                    from mycorrhizal.common.wrappers import create_view_from_protocol
                    from dataclasses import replace

                    # Check cache
                    cache_key = (id(context.common), interface_type)
                    if cache_key in _interface_view_cache:
                        constrained_common = _interface_view_cache[cache_key]
                        return replace(context, common=constrained_common)

                    # Create constrained view of common with interface metadata
                    readonly_fields = getattr(interface_type, '_readonly_fields', set())
                    constrained_common = create_view_from_protocol(
                        context.common,
                        interface_type,
                        readonly_fields=readonly_fields
                    )

                    # Cache for reuse
                    _interface_view_cache[cache_key] = constrained_common

                    # Return new context with constrained common
                    return replace(context, common=constrained_common)
    except Exception:
        # If anything goes wrong with type inspection, fall back to original context
        pass

    return context


@dataclass
class StateConfiguration:
    """Configuration options for states.

    Args:
        timeout: Optional timeout in seconds. If no message received within
            this time, on_timeout handler is called.
        retries: Optional number of retries allowed. If state returns Retry,
            the retry counter increments. When max retries exceeded, state fails.
        terminal: If True, reaching this state completes the FSM (raises
            StateMachineComplete exception).
        can_dwell: If True, state can wait indefinitely without returning a
            transition (returning None from on_state is allowed).

    Example:
        @enoki.state(config=StateConfiguration(timeout=5.0, retries=3))
        def MyState():
            # State with timeout and retry handling
            pass
    """

    timeout: Optional[float] = None
    retries: Optional[int] = None
    terminal: bool = False
    can_dwell: bool = False


@dataclass
class SharedContext(Generic[T]):
    """Context passed to state handler methods.

    The SharedContext provides access to the current message, shared data,
    and utilities for state handlers.

    Attributes:
        send_message: Function to send messages to the state machine
        log: Function to log messages (default: prints with [FSM] prefix)
        common: Shared data passed to StateMachine (accessible as ctx.common)
        msg: The current message being processed (if any)

    Type parameter T represents the type of the 'common' field for type safety.

    Example:
        @enoki.on_state
        async def on_state(ctx: SharedContext):
            # Access shared data
            counter = ctx.common.get("counter", 0)

            # Check for messages
            if ctx.msg == "start":
                return Events.START

            # Send new messages
            ctx.send_message("ping")
    """

    send_message: Callable
    log: Callable
    common: T
    msg: Optional[Any] = None


# ============================================================================
# Transition Type Hierarchy
# ============================================================================

@dataclass
class TransitionType:
    """Base type for all transitions"""

    AWAITS: bool = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def base_name(self) -> str:
        if hasattr(self.__class__, "__bases__") and self.__class__.__bases__:
            return self.__class__.__bases__[0].__name__
        return self.name

    def __hash__(self):
        return hash(f"{self.base_name()}.{self.name}")

    def __eq__(self, other):
        if isinstance(other, TransitionType):
            return self.name == other.name
        return False


@dataclass
class StateTransition(TransitionType):
    """Represents a transition to a different state"""

    pass


@dataclass
class StateContinuation(TransitionType):
    """Represents staying in the same state"""

    pass


@dataclass
class StateRenewal(TransitionType):
    """Represents a transition back into the current state"""


@dataclass
class Again(StateContinuation):
    """Execute current state immediately without affecting retry counter"""

    pass


@dataclass
class Unhandled(StateContinuation):
    """Wait for next message/event without affecting retry counter"""

    AWAITS: bool = True


@dataclass
class Retry(StateContinuation):
    """Retry current state, decrementing retry counter, starting from on_enter"""

    pass


@dataclass
class Restart(StateRenewal):
    """Restart current state, reset retry counter, await message"""

    AWAITS: bool = True


@dataclass
class Repeat(StateRenewal):
    """Repeat current state, reset retry counter, execute on_enter immediately"""

    pass


@dataclass
class StateRef(StateTransition):
    """String-based state reference for breaking circular imports"""

    state: str = ""

    def __init__(self, state: str):
        self.state = state
        object.__setattr__(self, "AWAITS", False)

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        if isinstance(other, StateRef):
            return self.state == other.state
        return False


@dataclass
class LabeledTransition:
    """A transition with a label (typically an enum value)"""

    label: Enum
    transition: TransitionType


# ============================================================================
# Push/Pop Transitions (defined later to depend on StateSpec)
# ============================================================================

@dataclass
class Push(StateTransition):
    """Push one or more states onto the stack"""

    push_states: List[StateTransition] = field(default_factory=list)

    def __init__(self, *states: StateTransition):
        object.__setattr__(self, "push_states", list(states))

    def __hash__(self):
        return hash(".".join([s.name for s in self.push_states]))


@dataclass
class Pop(StateTransition):
    """Pop from the stack to return to previous state"""

    pass


# ============================================================================
# StateSpec - The heart of the decorator system
# ============================================================================

@dataclass
class StateSpec(StateTransition):
    """
    A state defined via decorators.

    This is a StateTransition, so it can be used directly in transition lists.
    It has the same interface as the old State class but is constructed by
    decorators rather than metaclass magic.
    """

    # Core identity
    name: str = ""
    qualname: str = ""
    module: str = ""

    # Configuration
    config: StateConfiguration = field(default_factory=StateConfiguration)

    # Lifecycle methods (all optional except on_state)
    on_state: Optional[Callable[[SharedContext], Awaitable[TransitionType]]] = None
    on_enter: Optional[Callable[[SharedContext], Awaitable[None]]] = None
    on_leave: Optional[Callable[[SharedContext], Awaitable[None]]] = None
    on_timeout: Optional[Callable[[SharedContext], Awaitable[TransitionType]]] = None
    on_fail: Optional[Callable[[SharedContext], Awaitable[TransitionType]]] = None

    # Transitions
    transitions: Optional[Callable[[], List[Union[LabeledTransition, TransitionType]]]] = None

    # Nested event enum (optional)
    Events: Optional[type[Enum]] = None

    # Group information (for state groups/namespaces)
    group_name: str = field(default="")
    parent_state: Optional[StateSpec] = None

    # Metadata
    _is_root: bool = field(default=False, repr=False)

    @property
    def base_name(self) -> str:
        """Get just the state name without module path"""
        return self.qualname.split(".")[-1]

    @property
    def CONFIG(self) -> StateConfiguration:
        """Alias for config (for compatibility with State class API)"""
        return self.config

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, StateSpec):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def get_transitions(self) -> List[Union[LabeledTransition, TransitionType]]:
        """Get the transition list for this state"""
        if self.transitions is None:
            return []
        result = self.transitions()
        # Handle single transition returned
        if isinstance(result, (TransitionType, LabeledTransition)):
            return [result]
        return result


# ============================================================================
# Global Registry
# ============================================================================

_state_registry: Dict[str, StateSpec] = {}


def register_state(state: StateSpec) -> None:
    """Register a state in the global registry"""
    _state_registry[state.name] = state


def get_state(name: str) -> Optional[StateSpec]:
    """Get a state from the global registry"""
    return _state_registry.get(name)


def get_all_states() -> Dict[str, StateSpec]:
    """Get all registered states"""
    return _state_registry.copy()


# ============================================================================
# StateRegistry - Validation and Resolution
# ============================================================================

class ValidationError(Exception):
    """Raised when state machine validation fails"""
    pass


@dataclass
class ValidationResult:
    """Results of state machine validation"""
    valid: bool
    errors: list[str]
    warnings: list[str]
    discovered_states: set[str]


class StateRegistry:
    """
    Manages state resolution and validation for StateSpec-based states.

    This registry handles:
    - Resolving StateRef strings to StateSpec objects
    - Validating all states in a state machine
    - Getting transition mappings for states
    - Detecting circular references and invalid transitions
    """

    def __init__(self):
        self._state_cache: Dict[str, StateSpec] = {}
        self._transition_cache: Dict[str, Dict] = {}
        self._resolved_modules: Dict[str, Any] = {}
        self._validation_errors: list[str] = []
        self._validation_warnings: list[str] = []

    def resolve_state(
        self, state_ref: Union[str, StateRef, StateSpec], validate_only: bool = False
    ) -> Optional[StateSpec]:
        """
        Resolve a state reference to an actual StateSpec object.

        Args:
            state_ref: Can be a StateSpec, a StateRef, or a fully qualified state name string
            validate_only: If True, don't cache the result (used during validation)

        Returns:
            The resolved StateSpec, or None if not found (when validate_only=True)
        """
        # If it's already a StateSpec, return it
        if isinstance(state_ref, StateSpec):
            # Track resolved module
            if state_ref.module and state_ref.module not in self._resolved_modules:
                mod = sys.modules.get(state_ref.module)
                self._resolved_modules[state_ref.module] = mod

                # Store short name for easier local file lookup
                if mod and hasattr(mod, "__file__"):
                    fp = Path(getattr(mod, "__file__"))
                    self._resolved_modules[fp.stem] = mod

            return state_ref

        # If it's a StateRef, resolve the string
        if isinstance(state_ref, StateRef):
            state_name = state_ref.state
        elif isinstance(state_ref, str):
            state_name = state_ref
        else:
            return None

        # Check cache first
        if state_name in self._state_cache and not validate_only:
            return self._state_cache[state_name]

        # Try to get from global registry
        state = get_state(state_name)
        if state:
            if not validate_only:
                self._state_cache[state_name] = state
            return state

        # If not in global registry, we can't resolve it
        # (In the old system, it would try dynamic imports, but with
        # decorator-based states, everything should be pre-registered)
        if validate_only:
            self._validation_errors.append(
                f"State '{state_name}' not found in registry"
            )
            return None
        else:
            raise ValidationError(f"State '{state_name}' not found in registry")

    def validate_all_states(
        self,
        initial_state: Union[str, StateRef, StateSpec],
        error_state: Optional[Union[str, StateRef, StateSpec]] = None,
    ) -> ValidationResult:
        """
        Validate all states reachable from the initial state.

        Args:
            initial_state: The starting state
            error_state: Optional error state

        Returns:
            ValidationResult with errors, warnings, and discovered states
        """
        self._validation_errors = []
        self._validation_warnings = []

        # Discover all reachable states
        discovered_states = set()
        state_references = set()
        resolved_references = set()
        states_to_visit = []

        # Resolve initial state
        initial_resolved = self.resolve_state(initial_state, validate_only=True)
        if initial_resolved:
            states_to_visit.append(initial_resolved)
            discovered_states.add(initial_resolved.name)

        # Resolve error state if provided
        if error_state:
            error_resolved = self.resolve_state(error_state, validate_only=True)
            if error_resolved:
                states_to_visit.append(error_resolved)
                discovered_states.add(error_resolved.name)

        # Traverse all reachable states
        visited = set()
        while states_to_visit:
            current_state = states_to_visit.pop()
            state_name = current_state.name

            if state_name in visited:
                continue
            visited.add(state_name)

            def handle_state(s):
                discovered_states.add(s.name if hasattr(s, 'name') else str(s))
                states_to_visit.append(s)

            def handle_reference(r):
                state_references.add(r)

            def handle_push(p):
                for state in p.push_states:
                    match state:
                        case s if isinstance(s, StateSpec):
                            handle_state(s)
                        case r if isinstance(r, StateRef):
                            handle_reference(r)
                        case invalid:
                            self._validation_errors.append(
                                f"State '{state_name}' attempted to push an invalid transition: {invalid}"
                            )

            try:
                # Get transitions
                raw_transitions = current_state.get_transitions()

                # Handle different return types
                match raw_transitions:
                    case list() | tuple():
                        pass  # Already a list
                    case _ if isinstance(raw_transitions, (TransitionType, LabeledTransition)):
                        raw_transitions = [raw_transitions]
                    case _:
                        raw_transitions = []

                # Validate each transition
                for transition in raw_transitions:
                    match transition:
                        case _ if isinstance(transition, Push):
                            # Push transitions must have labels
                            self._validation_errors.append(
                                f"State '{state_name}' has a push transition without a label"
                            )
                            handle_push(transition)
                        case LabeledTransition(label, s):
                            match s:
                                case st if isinstance(st, StateSpec):
                                    handle_state(st)
                                case r if isinstance(r, StateRef):
                                    handle_reference(r)
                                case p if isinstance(p, Push):
                                    handle_push(p)
                        case s if isinstance(s, StateSpec):
                            handle_state(s)
                        case r if isinstance(r, StateRef):
                            self._validation_warnings.append(
                                f"State '{state_name}' has a StateReference transition without a label"
                            )
                            handle_reference(r)
                        case _ if isinstance(transition, (StateContinuation, StateRenewal)):
                            # Valid continuation/renewal
                            pass
                        case invalid:
                            self._validation_errors.append(
                                f"State '{state_name}' has an invalid transition: {invalid}"
                            )

                # Validate that the state has an on_state handler
                if current_state.on_state is None:
                    self._validation_errors.append(
                        f"State {state_name} does not have an on_state handler defined"
                    )

                # Make sure non-terminal states define transitions
                if not current_state.config.terminal:
                    if current_state.transitions is None:
                        self._validation_errors.append(
                            f"Non-terminal state {state_name} does not define any transitions"
                        )

                # Validate timeout configuration
                if current_state.config.timeout is not None:
                    if (
                        not isinstance(current_state.config.timeout, (int, float))
                        or current_state.config.timeout <= 0
                    ):
                        self._validation_errors.append(
                            f"State '{state_name}' has invalid timeout: {current_state.config.timeout}"
                        )

                    has_timeout_handler = current_state.on_timeout is not None
                    if not has_timeout_handler:
                        self._validation_warnings.append(
                            f"State '{state_name}' has timeout but no on_timeout handler defined"
                        )

                # Pass 2: Validate all string references
                for state_ref in list(state_references - resolved_references):
                    resolved = self.resolve_state(state_ref, validate_only=True)
                    if resolved:
                        resolved_references.add(state_ref)
                        if resolved.name not in discovered_states:
                            discovered_states.add(resolved.name)
                            states_to_visit.append(resolved)

            except Exception as e:
                self._validation_errors.append(
                    f"Error processing transitions for state '{state_name}': {traceback.format_exc()}"
                )

        return ValidationResult(
            valid=len(self._validation_errors) == 0,
            errors=self._validation_errors.copy(),
            warnings=self._validation_warnings.copy(),
            discovered_states=discovered_states,
        )

    def get_transitions(self, state: StateSpec) -> Dict:
        """
        Get the transition mapping for a state, with caching.

        Returns a dict mapping event labels/enum values to transition targets.
        """
        state_name = state.name

        if state_name in self._transition_cache:
            return self._transition_cache[state_name]

        resolved_transitions = {}

        raw_transitions = state.get_transitions()
        match raw_transitions:
            case list() | tuple():
                pass  # Already a list
            case _ if isinstance(raw_transitions, (TransitionType, LabeledTransition)):
                raw_transitions = [raw_transitions]
            case _:
                raw_transitions = []

        for transition in raw_transitions:
            match transition:
                case LabeledTransition(label, target):
                    # Use the label enum itself as the key
                    resolved_transitions[label] = target
                    # Also allow lookup by label name
                    resolved_transitions[label.name] = target
                    # Also allow lookup by label value
                    if hasattr(label, 'value'):
                        resolved_transitions[label.value] = target
                case s if isinstance(s, StateSpec):
                    # Direct state reference
                    resolved_transitions[s.name] = s
                    resolved_transitions[s] = s
                case r if isinstance(r, StateRef):
                    # String reference - resolve it
                    resolved = self.resolve_state(r)
                    if resolved:
                        resolved_transitions[r.state] = resolved
                case _ if isinstance(transition, (StateContinuation, StateRenewal)):
                    # Special transitions - allow lookup by type
                    resolved_transitions[transition] = transition

        self._transition_cache[state_name] = resolved_transitions
        return resolved_transitions


# ============================================================================
# Message Types for StateMachine
# ============================================================================

@dataclass
class PrioritizedMessage:
    priority: int
    item: Any = field(compare=False)

    def __lt__(self, other):
        """Compare based on priority only"""
        if not isinstance(other, PrioritizedMessage):
            return NotImplemented
        return self.priority < other.priority


@dataclass
class EnokiInternalMessage:
    """Base class for internal FSM messages"""
    pass


@dataclass
class TimeoutMessage(EnokiInternalMessage):
    """Internal message for timeout events"""
    state_name: str
    timeout_id: int
    timeout_duration: float


# ============================================================================
# Exceptions for StateMachine
# ============================================================================

class StateMachineComplete(Exception):
    """Raised when the state machine has reached a terminal state"""
    pass


class BlockedInUntimedState(Exception):
    """Raised when a non-dwelling state blocks without a timeout"""
    pass


class PopFromEmptyStack(Exception):
    """Raised when we attempt to pop from an empty stack"""
    pass


class NoStateToTick(Exception):
    """Raised when the FSM object doesn't have a state attached"""
    pass


# ============================================================================
# StateMachine Runtime
# ============================================================================

class StateMachine:
    """Asyncio-native finite state machine for executing StateSpec-based states.

    The StateMachine manages state execution, transitions, message passing,
    and lifecycle handling. States are defined using the @enoki.state decorator
    and should define on_state, on_enter, on_leave, on_timeout, or on_fail handlers.

    Args:
        initial_state: The initial state (state function, StateRef, or StateSpec)
        error_state: Optional error state for handling exceptions
        filter_fn: Optional function to filter messages (ctx, msg) -> bool
        trap_fn: Optional function to trap exceptions (exc) -> None
        on_error_fn: Optional function called on errors (ctx, exc) -> None
        common_data: Shared data accessible via ctx.common in all states

    Attributes:
        current_state: The currently executing state
        context: SharedContext containing msg and common data
        state_stack: Stack for Push/Pop hierarchical state management

    Methods:
        initialize(): Initialize the state machine (must be called before tick)
        tick(timeout): Process one state machine tick
        send_message(msg): Send a message to the state machine
        reset(): Reset to initial state

    Example:
        fsm = StateMachine(
            initial_state=IdleState,
            error_state=ErrorState,
            common_data={"counter": 0}
        )
        await fsm.initialize()
        fsm.send_message("start")
        await fsm.tick()
    """

    class MessagePriorities(IntEnum):
        ERROR = 0
        INTERNAL_MESSAGE = 1
        MESSAGE = 2

    def __init__(
        self,
        initial_state: Union[str, StateRef, StateSpec],
        error_state: Optional[Union[str, StateRef, StateSpec]] = None,
        filter_fn: Optional[Callable] = None,
        trap_fn: Optional[Callable] = None,
        on_error_fn: Optional[Callable] = None,
        common_data: Optional[Any] = None,
    ):

        self.registry = StateRegistry()

        # The filter and trap functions
        self._filter_fn = filter_fn or (lambda x, y: None)
        self._trap_fn = trap_fn or (lambda x: None)
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

        # Resolve initial and error states
        self.initial_state = self.registry.resolve_state(initial_state)
        self.error_state = (
            self.registry.resolve_state(error_state) if error_state else None
        )

        # Asyncio-based infrastructure
        self._message_queue = PriorityQueue()
        self._timeout_task = None
        self._timeout_counter = 0
        self._current_timeout_id = None

        # Runtime state
        self.current_state: Optional[StateSpec] = None
        self.state_stack: List[StateSpec] = []
        self.context = SharedContext(
            send_message=self.send_message,
            log=self.log,
            common=common_data if common_data is not None else dict(),
        )

        self.retry_counters: Dict[str, int] = {}
        self.state_enter_times: Dict[str, float] = {}

        # Initialize asynchronously
        self._initialized = False

    async def initialize(self):
        """Initialize the state machine asynchronously"""
        if not self._initialized:
            await self.reset()
            self._initialized = True

    def send_message(self, message: Any):
        """Send a message to the state machine"""
        try:
            if message and self._filter_fn(self.context, message):
                return
            self._send_message_internal(message)
        except Exception as e:
            self._send_message_internal(e)

    def _send_message_internal(self, message: Any):
        """Internal message sending"""

        match message:
            case _ if isinstance(message, EnokiInternalMessage):
                try:
                    self._message_queue.put_nowait(
                        PrioritizedMessage(
                            self.MessagePriorities.INTERNAL_MESSAGE, message
                        )
                    )
                except Exception:
                    pass  # Queue full
            case _ if isinstance(message, Exception):
                try:
                    self._message_queue.put_nowait(
                        PrioritizedMessage(self.MessagePriorities.ERROR, message)
                    )
                except Exception:
                    pass
            case _:
                try:
                    self._message_queue.put_nowait(
                        PrioritizedMessage(self.MessagePriorities.MESSAGE, message)
                    )
                except Exception:
                    pass

    async def reset(self):
        """Reset the state machine to initial state"""
        self._cancel_timeout()
        self.state_stack = []
        self.retry_counters = {}
        self.state_enter_times = {}

        # Clear message queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except Exception:
                break

        # Transition into our initial state
        await self._transition_to_state(self.initial_state)

    def log(self, message: str):
        """Log a message"""
        print(f"[FSM] {message}")

    async def tick(self, timeout: Optional[Union[float, int]] = 0):
        """Process one state machine tick"""

        if not self._initialized:
            await self.initialize()

        if not self.current_state:
            raise NoStateToTick()

        # Validate timeout parameter
        match timeout:
            case x if x and not isinstance(x, (int, float)):
                raise ValueError(f"Tick timeout must be None, or an int/float >= 0")
            case 0:
                block = False
            case None:
                block = True
            case x if x > 0:
                block = True
            case _:
                raise ValueError("Tick timeout must be None or >= 0")

        while True:
            try:
                if isinstance(self.context.msg, Exception):
                    raise self.context.msg

                # Handle timeout messages specially
                if isinstance(self.context.msg, TimeoutMessage):
                    if not self._handle_timeout_message():
                        return
                    transition = await self._call_on_timeout(self.current_state)
                else:
                    transition = await self._call_on_state(self.current_state)

                if self.current_state.config.terminal:
                    raise StateMachineComplete()

                if transition in (None, Unhandled):
                    self._trap_fn(self.context)

                self.context.msg = None

                should_check_queue = await self._process_transition(transition, block)

                # In manual mode (block=False), stop after processing one transition if not checking queue
                if not block and not should_check_queue:
                    break

                if should_check_queue:
                    if block:
                        message = await asyncio.wait_for(
                            self._message_queue.get(), timeout=timeout
                        )
                        message = message.item
                    else:
                        message = self._message_queue.get_nowait().item
                    self.context.msg = message
            except asyncio.QueueEmpty:
                break
            except asyncio.TimeoutError:
                break
            except Exception as e:
                if (
                    isinstance(e, StateMachineComplete)
                    or self.current_state.config.terminal
                ):
                    raise

                next_transition = self._on_err_fn(self.context, e)
                if next_transition:
                    await self._process_transition(next_transition, block)
                elif self.error_state and self.current_state != self.error_state:
                    # Clear the exception message before transitioning
                    self.context.msg = None
                    await self._transition_to_state(self.error_state)
                elif self.current_state == self.error_state:
                    # Already in error state and got another exception - stop
                    break
                else:
                    raise

        return None

    async def _call_on_state(self, state: StateSpec):
        """Call the state's on_state handler"""
        if state.on_state is None:
            raise ValueError(f"State {state.name} has no on_state handler")
        # Create interface view if handler has interface type hint
        ctx_to_pass = _create_interface_view_for_context(self.context, state.on_state)
        return await state.on_state(ctx_to_pass)

    async def _call_on_enter(self, state: StateSpec):
        """Call the state's on_enter handler"""
        if state.on_enter is not None:
            self.log(f"  [DEBUG] Calling on_enter for {state.name}")
            # Create interface view if handler has interface type hint
            ctx_to_pass = _create_interface_view_for_context(self.context, state.on_enter)
            await state.on_enter(ctx_to_pass)
            self.log(f"  [DEBUG] on_enter completed for {state.name}")

    async def _call_on_leave(self, state: StateSpec):
        """Call the state's on_leave handler"""
        if state.on_leave is not None:
            # Create interface view if handler has interface type hint
            ctx_to_pass = _create_interface_view_for_context(self.context, state.on_leave)
            await state.on_leave(ctx_to_pass)

    async def _call_on_timeout(self, state: StateSpec):
        """Call the state's on_timeout handler"""
        if state.on_timeout is None:
            raise ValueError(f"State {state.name} has no on_timeout handler")
        # Create interface view if handler has interface type hint
        ctx_to_pass = _create_interface_view_for_context(self.context, state.on_timeout)
        return await state.on_timeout(ctx_to_pass)

    async def _call_on_fail(self, state: StateSpec):
        """Call the state's on_fail handler"""
        if state.on_fail is None:
            raise ValueError(f"State {state.name} has no on_fail handler")
        # Create interface view if handler has interface type hint
        ctx_to_pass = _create_interface_view_for_context(self.context, state.on_fail)
        return await state.on_fail(ctx_to_pass)

    async def _process_transition(self, transition: Any, block: bool = False) -> bool:
        """Process a state transition"""

        valid_transitions = self.registry.get_transitions(self.current_state)

        # Try to get the transition target
        target = valid_transitions.get(transition)

        # If not found by the transition itself, try by value
        if target is None and transition in valid_transitions.values():
            target = transition
        elif transition is not None and target is None:
            raise ValueError(f"Unknown transition {transition}: {valid_transitions}")

        # Process the transition

        if transition in (None, Unhandled):
            if (
                self.current_state.config.can_dwell
            ) or self.current_state.config.timeout:
                return True
            raise BlockedInUntimedState(
                f"{self.current_state.name} cannot dwell and does not have a timeout"
            )
        elif isinstance(target, StateSpec):
            await self._transition_to_state(target)
        elif isinstance(target, Push):
            await self._handle_push(target)
        elif isinstance(target, Pop) or target is Pop:
            await self._handle_pop()
        elif target == Again:
            # Re-execute current state immediately
            # In manual tick mode (block=False), stop after one execution
            # In automatic mode (block=True), continue the loop
            if block:
                # Automatic mode: continue the loop
                return True
            else:
                # Manual mode: stop after this execution
                return False
        elif target == Repeat:
            await self._transition_to_state(self.current_state)
        elif target == Restart:
            self._reset_state_context()
            await self._transition_to_state(self.current_state)
            return True
        elif target == Retry:
            await self._handle_retry()
        elif isinstance(target, (StateContinuation, StateRenewal)):
            # Other continuations/renewals - handle appropriately
            if target == Again:
                return True  # Continue the loop
            elif target == Repeat:
                await self._transition_to_state(self.current_state)
            elif target == Restart:
                self._reset_state_context()
                await self._transition_to_state(self.current_state)
                return True
            elif target == Retry:
                await self._handle_retry()

        return False

    async def _transition_to_state(self, next_state: StateSpec):
        """Transition to a new state"""

        if self.current_state:
            await self._call_on_leave(self.current_state)
            self._cancel_timeout()

        self.current_state = next_state
        self.state_enter_times[next_state.name] = time.time()
        self.log(f"Transitioned to {next_state.name}")
        self.log(f"  [DEBUG] common object id: {id(self.context.common)}")
        self.log(f"  [DEBUG] on_enter is None? {next_state.on_enter is None}")

        self._start_timeout(next_state)
        await self._call_on_enter(next_state)
        self.log(f"  [DEBUG] After on_enter, common: {self.context.common}")

    def _start_timeout(self, state: StateSpec) -> Optional[int]:
        """Start a timeout for the given state"""

        if state.config.timeout is None:
            return None

        self._cancel_timeout()

        self._timeout_counter += 1
        timeout_id = self._timeout_counter
        self._current_timeout_id = timeout_id

        async def timeout_handler():
            try:
                await asyncio.sleep(state.config.timeout)
                self._send_timeout_message(state.name, timeout_id, state.config.timeout)
            except asyncio.CancelledError:
                pass

        self._timeout_task = asyncio.create_task(timeout_handler())
        return timeout_id

    def _cancel_timeout(self):
        """Cancel any active timeout"""
        if self._timeout_task is not None:
            self._timeout_task.cancel()
            self._timeout_task = None
        self._current_timeout_id = None

    def _send_timeout_message(self, state_name: str, timeout_id: int, duration: float):
        """Internal method to send timeout messages"""

        timeout_msg = TimeoutMessage(state_name, timeout_id, duration)
        self._send_message_internal(timeout_msg)

    def _handle_timeout_message(self) -> bool:
        """Handle a timeout message"""

        timeout_msg = self.context.msg
        self.context.msg = None

        if timeout_msg.timeout_id != self._current_timeout_id:
            self.log(f"Ignoring stale timeout {timeout_msg.timeout_id}")
            return False

        if timeout_msg.state_name != self.current_state.name:
            self.log(
                f"Ignoring timeout for {timeout_msg.state_name}, current state is {self.current_state.name}"
            )
            return False

        self.log(
            f"Handling timeout for {timeout_msg.state_name} ({timeout_msg.timeout_duration}s)"
        )
        self._cancel_timeout()
        return True

    async def _handle_retry(self):
        """Handle retry logic with counter"""
        state_name = self.current_state.name

        if state_name not in self.retry_counters:
            self.retry_counters[state_name] = 0

        self.retry_counters[state_name] += 1

        if (
            self.current_state.config.retries is not None
            and self.retry_counters[state_name] > self.current_state.config.retries
        ):
            self.log(f"Retry limit exceeded for {state_name}")
            transition = await self._call_on_fail(self.current_state)
            await self._process_transition(transition)
        else:
            await self._transition_to_state(self.current_state)

    def _reset_state_context(self):
        """Reset context for current state"""
        state_name = self.current_state.name
        if state_name in self.retry_counters:
            del self.retry_counters[state_name]

    async def _handle_push(self, push: Push):
        """Handle push transition"""
        resolved_states = [self.registry.resolve_state(s) for s in push.push_states]

        for state in reversed(resolved_states[1:]):
            self.state_stack.append(state)

        await self._transition_to_state(resolved_states[0])

    async def _handle_pop(self):
        """Handle pop transition"""
        if not self.state_stack:
            raise PopFromEmptyStack()

        next_state = self.state_stack.pop()
        await self._transition_to_state(next_state)

    async def run(
        self, max_iterations: Optional[int] = None, timeout: Optional[float] = 1
    ):
        """Run the state machine until terminal state"""
        if not self._initialized:
            await self.initialize()

        iteration = 0

        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break

            if self.current_state.config.terminal:
                self.log(f"Reached terminal state: {self.current_state.name}")
                break

            try:
                await self.tick(timeout=timeout)
                iteration += 1
            except StateMachineComplete:
                raise
            except Exception as e:
                self.log(f"Error in FSM: {e}")
                if self.error_state:
                    await self._transition_to_state(self.error_state)
                else:
                    raise


# ============================================================================
# Decorators
# ============================================================================

class _EnokiDecoratorAPI:
    """Decorator API for defining states"""

    def __init__(self):
        self._tracking_stack: List[List[Tuple[str, Any]]] = []

    def state(
        self,
        config: StateConfiguration = StateConfiguration(),
        name: Optional[str] = None,
        group: Optional[str] = None,
    ) -> Callable[[Callable], StateSpec]:
        """
        Decorator to define a state.

        Args:
            config: StateConfiguration for this state
            name: Optional name (auto-generated if not provided)
            group: Optional group name for organization

        The decorated function should contain nested decorated functions
        (@on_state, @on_enter, @transitions, etc.). No need to return a dict!
        """

        def decorator(func: Callable[..., Any]) -> StateSpec:
            # Get function metadata
            module = func.__module__
            qualname = func.__qualname__

            # Auto-generate name if not provided
            if name is None:
                # Handle __main__ module
                if module == "__main__":
                    filename = inspect.getfile(func)
                    module = os.path.splitext(os.path.basename(filename))[0]
                state_name = f"{module}.{qualname}"
            else:
                state_name = name

            # Set up tracking for this state
            tracked_items = []
            self._tracking_stack.append(tracked_items)

            try:
                # Call the function to execute inner decorators
                func()
            finally:
                # Always pop the tracking stack
                self._tracking_stack.pop()

            # Extract decorated methods from tracked items
            on_state = None
            on_enter = None
            on_leave = None
            on_timeout = None
            on_fail = None
            transitions = None
            Events = None

            for item_name, item_fn in tracked_items:
                if hasattr(item_fn, "_enoki_on_state"):
                    on_state = item_fn
                elif hasattr(item_fn, "_enoki_on_enter"):
                    on_enter = item_fn
                elif hasattr(item_fn, "_enoki_on_leave"):
                    on_leave = item_fn
                elif hasattr(item_fn, "_enoki_on_timeout"):
                    on_timeout = item_fn
                elif hasattr(item_fn, "_enoki_on_fail"):
                    on_fail = item_fn
                elif hasattr(item_fn, "_enoki_transitions"):
                    transitions = item_fn
                elif hasattr(item_fn, "_enoki_events"):
                    Events = item_fn

            # Validate required methods
            if on_state is None:
                raise ValueError(f"State '{state_name}' must have an @on_state decorated method")

            # Create the StateSpec
            state_spec = StateSpec(
                name=state_name,
                qualname=qualname,
                module=module,
                config=config,
                on_state=on_state,
                on_enter=on_enter,
                on_leave=on_leave,
                on_timeout=on_timeout,
                on_fail=on_fail,
                transitions=transitions,
                Events=Events,
                group_name=group or "",
            )

            # Register the state
            register_state(state_spec)

            return state_spec

        return decorator

    def on_state(self, func: Callable) -> Callable:
        """Decorator for the main state logic method"""
        func._enoki_on_state = func
        if self._tracking_stack:
            self._tracking_stack[-1].append((func.__name__, func))
        return func

    def on_enter(self, func: Callable) -> Callable:
        """Decorator for on_enter lifecycle method"""
        func._enoki_on_enter = func
        if self._tracking_stack:
            self._tracking_stack[-1].append((func.__name__, func))
        return func

    def on_leave(self, func: Callable) -> Callable:
        """Decorator for on_leave lifecycle method"""
        func._enoki_on_leave = func
        if self._tracking_stack:
            self._tracking_stack[-1].append((func.__name__, func))
        return func

    def on_timeout(self, func: Callable) -> Callable:
        """Decorator for on_timeout lifecycle method"""
        func._enoki_on_timeout = func
        if self._tracking_stack:
            self._tracking_stack[-1].append((func.__name__, func))
        return func

    def on_fail(self, func: Callable) -> Callable:
        """Decorator for on_fail lifecycle method"""
        func._enoki_on_fail = func
        if self._tracking_stack:
            self._tracking_stack[-1].append((func.__name__, func))
        return func

    def transitions(self, func: Callable) -> Callable:
        """Decorator for declaring transitions"""
        func._enoki_transitions = func
        if self._tracking_stack:
            self._tracking_stack[-1].append((func.__name__, func))
        return func

    def events(self, events_class: type) -> type:
        """
        Decorator for registering the Events enum.

        Optional decorator that makes the Events enum accessible via state.Events
        for introspection and testing. The Events enum works via closure even
        without this decorator, but state.Events will be None.

        Use this decorator when:
        - You need to access state.Events programmatically (e.g., in tests)
        - You want to inspect what events a state defines
        - Registry methods need to reference the Events enum

        Skip this decorator when:
        - You only use Events within on_state/transitions (closure handles it)
        - You don't need external access to the Events enum

        Example:
            @enoki.state()
            def MyState():
                @enoki.events  # Optional - makes MyState.Events accessible
                class Events(Enum):
                    GO = auto()

                @enoki.on_state
                async def on_state(ctx):
                    return Events.GO  # Works via closure even without @enoki.events
        """
        events_class._enoki_events = events_class
        if self._tracking_stack:
            self._tracking_stack[-1].append(("Events", events_class))
        return events_class

    def root(self, func: Callable) -> Callable:
        """Decorator to mark a state as the initial/root state"""
        func._enoki_is_root = True
        return func


# Create the decorator API instance
enoki = _EnokiDecoratorAPI()

# Export decorators for convenience
state = enoki.state
on_state = enoki.on_state
on_enter = enoki.on_enter
on_leave = enoki.on_leave
on_timeout = enoki.on_timeout
on_fail = enoki.on_fail
transitions = enoki.transitions
events = enoki.events
root = enoki.root


# ============================================================================
# Export all public components
# ============================================================================

__all__ = [
    # Decorators
    "enoki",
    "state",
    "on_state",
    "on_enter",
    "on_leave",
    "on_timeout",
    "on_fail",
    "transitions",
    "root",
    # Core types
    "StateSpec",
    "StateConfiguration",
    "SharedContext",
    # Transitions
    "TransitionType",
    "StateTransition",
    "StateContinuation",
    "StateRenewal",
    "Again",
    "Unhandled",
    "Retry",
    "Restart",
    "Repeat",
    "StateRef",
    "LabeledTransition",
    "Push",
    "Pop",
    # Registry
    "register_state",
    "get_state",
    "get_all_states",
    "StateRegistry",
    "ValidationError",
    "ValidationResult",
    # StateMachine
    "StateMachine",
    "StateMachineComplete",
    "BlockedInUntimedState",
    "PopFromEmptyStack",
    "NoStateToTick",
    # Messages
    "PrioritizedMessage",
    "EnokiInternalMessage",
    "TimeoutMessage",
]
