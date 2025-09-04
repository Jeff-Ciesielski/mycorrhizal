#!/usr/bin/env python3
"""
Hypha Core - Asynchronous Petri Net Library

Ported from gevent to asyncio for autonomous execution without colored functions.
Each transition runs as an independent asyncio task waiting on input queues.
"""

import asyncio
from asyncio import (
    Queue,
    QueueEmpty,
    QueueFull,
    Event,
    Task,
    create_task,
    sleep,
    wait_for,
    gather,
    wait_for,
    FIRST_COMPLETED,
    CancelledError,
)
import inspect
import traceback
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
    Callable,
    Tuple,
    Iterable,
)
from enum import Enum
from itertools import combinations, product
from functools import reduce
import sys
from contextvars import ContextVar

# A per-task stack of currently-building declarative namespaces (innermost at the end)
_DECL_STACK: ContextVar[tuple] = ContextVar("_DECL_STACK", default=())


def _decl_stack_get() -> list:
    """Return a copy of the current declarative-namespace stack (outer..inner)."""
    return list(_DECL_STACK.get())


def _decl_stack_push(ns: dict) -> None:
    stack = list(_DECL_STACK.get())
    stack.append(ns)
    _DECL_STACK.set(tuple(stack))


def _decl_stack_pop(ns: dict) -> None:
    stack = list(_DECL_STACK.get())
    if stack and stack[-1] is ns:
        stack.pop()
    else:
        # Fallback: remove first matching (handles odd nesting/cross-calls defensively)
        for i in range(len(stack) - 1, -1, -1):
            if stack[i] is ns:
                del stack[i]
                break
    _DECL_STACK.set(tuple(stack))


class _DeclarativeNamespace(dict):
    """
    Namespace mapping returned by DeclarativeMeta.__prepare__.

    It allows helpers to (a) recognize "we are inside a declarative class body",
    and (b) inject attributes into the *actual* class namespace.
    """

    __slots__ = ("__declarative_meta__",)

    def __init__(self, meta):
        super().__init__()
        # Not a class attribute; just a marker on the mapping object itself
        self.__declarative_meta__ = meta


class PlaceName(Enum):
    """Base class for place name enumerations - enforces IDE type checking"""

    pass


# Core Exceptions
class PetriNetComplete(Exception):
    """Raised when the Petri net reaches a terminal condition."""

    pass


class InvalidTokenType(Exception):
    """Raised when a token type doesn't match place expectations."""

    pass


class InsufficientTokens(Exception):
    """Raised when a transition can't fire due to missing tokens."""

    pass


class InvalidPlaceType(Exception):
    """Raised when a place type is invalid."""

    pass


class InvalidTransitionType(Exception):
    """Raised when a transition type is invalid."""

    pass


class ArcValidationError(Exception):
    """Raised when arc definitions are invalid."""

    pass


class InterfaceBoundaryViolation(Exception):
    """Raised when a transition tries to access places outside its interface boundary."""

    pass


class MissingPlaceError(Exception):
    """Raised when a transition references a place that does not exist in the net."""

    pass


class QualifiedNameConflict(Exception):
    """Raised when there are conflicts in qualified names."""

    pass


class PetriNetDeadlock(Exception):
    """Raised when the Petri net is deadlocked."""

    pass


class Place(ABC):
    """
    Base class for all places in the Petri net system.

    Each Place can only act as input to a single transition
    """

    MAX_CAPACITY: Optional[int] = None
    ACCEPTED_TOKEN_TYPES: Optional[Set[Type[Any]]] = None

    def __init__(self):
        self._tokens: List[Any] = []
        self._total_tokens_processed = 0
        self._net: Optional["PetriNet"] = None
        self._qualified_name: Optional[str] = None
        self._task: Optional[Task] = None
        self._token_pulse: Event = Event()
        self._has_capacity: Event = Event()
        self._not_empty: Event = Event()

        self._has_capacity.set()

    def _attach_to_net(self, net: "PetriNet"):
        """Called when place is attached to a net."""
        self._net = net

    def _set_qualified_name(self, qualified_name: str):
        """Set the fully qualified name for this place."""
        self._qualified_name = qualified_name

    @property
    def net(self) -> "PetriNet":
        if not self._net:
            raise ValueError("Place is not attached to a Petri net.")
        return self._net

    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name for this place."""
        return self._qualified_name or self.__class__.__name__

    @property
    def token_count(self) -> int:
        """Returns the current number of tokens in this place."""
        return len(self._tokens)

    @property
    def is_empty(self) -> bool:
        """Returns True if the place has no tokens."""
        return self.token_count == 0

    @property
    def is_full(self) -> bool:
        """Returns True if the place is at maximum capacity."""
        if self.MAX_CAPACITY is None:
            return False
        return self.token_count >= self.MAX_CAPACITY

    def can_accept_token(self, token: Any) -> bool:
        """Check if this place can accept the given token."""
        if self.is_full:
            return False

        if self.ACCEPTED_TOKEN_TYPES is not None:
            return type(token) in self.ACCEPTED_TOKEN_TYPES

        return True

    async def wait_for_tokens(self):
        await self._token_pulse.wait()
        self._token_pulse.clear()

    async def add_token(self, token: Any, timeout: Optional[float] = None):
        """
        Add a token to this place's queue.

        Args:
            token: Token to add
            timeout: Timeout for blocking operation

        """
        while True:
            await wait_for(self._has_capacity.wait(), timeout)
            # If we have multiple writers waiting, the first will await, then clear this second barrier,
            # Subsequent writers will keep waiting their turn
            if self._has_capacity.is_set():
                break
        self._tokens.append(token)
        if self.MAX_CAPACITY is not None and self.token_count >= self.MAX_CAPACITY:
            self._has_capacity.clear()

        self._total_tokens_processed += 1
        await self.on_token_added(token)
        self._not_empty.set()  # Signal that we have tokens now
        self._token_pulse.set()
        self._token_pulse.clear()

    def remove_token(self, token: Any):
        """Remove a token from this place's bag."""
        self._tokens.remove(token)
        if self.MAX_CAPACITY is not None and self.token_count < self.MAX_CAPACITY:
            self._has_capacity.set()
        if self.token_count == 0:
            self._not_empty.clear()

    def combinations(self, r: int) -> Iterable[Tuple[Any, ...]]:
        """Get all combinations of tokens in this place."""
        return combinations(self._tokens, r)

    async def on_token_added(self, token: Any):
        """Called when a token is added to this place."""
        pass

    async def on_token_removed(self, token: Any):
        """Called when a token is removed from this place."""
        pass

    def __repr__(self):
        return f"{self.qualified_name}(tokens={self.token_count})"


class IOOutputPlace(Place):
    """
    Output IO places that send tokens to external systems.

    These places sink tokens produced by transitions and handle them autonomously.
    """

    async def add_token(self, token: Any, timeout: float | None = None):
        # Asynchronously handle the token addition as the on_token_added method may
        # perform IO operations that should not block the main event loop.
        async def _sink():
            await super(IOOutputPlace, self).add_token(token, timeout)
            self.remove_token(token)

        self.net.add_task_fn(_sink)

    async def on_token_added(self, token: Any):
        """
        Handle outgoing token. Override this for output places.
        """
        # Default implementation just logs and succeeds
        await self.log(f"IO Output: {token}")
        return None

    async def on_output_start(self):
        """Called when output processing starts."""
        pass

    async def on_output_stop(self):
        """Called when output processing stops."""
        pass

    async def on_io_success(self, token: Any):
        """Called when IO operation succeeds."""
        pass

    async def on_io_error(self, token: Any, error_token: Optional[Any]):
        """Called when IO operation fails."""
        pass

    async def log(self, message: str):
        """Log a message."""
        if self._net:
            await self._net.log(message)
        else:
            print(message)


class IOInputPlace(Place):
    """
    Input IO places that receive tokens from external systems.

    These places run autonomous tasks that produce tokens into their queues
    from external events/sources.
    """

    def _start_input_task(self):
        """Start the input processing task."""
        if self._task is None:
            self._task = self.net.add_task_fn(self._input_loop)

    async def _input_loop(self):
        """Main input loop - calls on_input() and puts tokens in queue."""
        try:
            await self.on_input_start()
            while True:
                # Call user-defined input handler
                result = await self.on_input()

                if result is None:
                    pass
                elif isinstance(result, (list, tuple)):
                    for token in result:
                        await self.add_token(token)
                        await self.on_token_added(token)
                else:
                    await self.add_token(result)
                    await self.on_token_added(result)

                # Yield control so other tasks can run
                await sleep(0)
        except CancelledError:
            await self.on_input_stop()
        finally:
            if self._net and self._task in self._net._running_tasks:
                self._net._running_tasks.discard(self._task)

    async def on_input_start(self):
        """Called when input starts."""
        pass

    async def on_input(self) -> Union[Any, List[Any], None]:
        """
        Main input handler - override this to generate tokens.

        This method is called repeatedly until shutdown.
        Return tokens to add to this place's queue.
        """
        # Default implementation waits for shutdown
        await sleep(1.0)
        return None

    async def on_input_stop(self):
        """Called when input stops."""
        pass

    async def log(self, message: str):
        """Log a message."""
        if self._net:
            await self._net.log(message)
        else:
            print(message)


@dataclass
class Arc:
    """Represents a connection between a place and transition with metadata."""

    place: Union[Type[Place], str]  # Place reference - class or qualified name string
    weight: int = 1
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            if isinstance(self.place, str):
                self.label = self.place.split(".")[-1].lower()
            elif self.place is not None:
                self.label = self.place.__name__.lower()
            else:
                self.label = "unnamed_arc"


# Core Transition System
class Transition(ABC):
    """
    Base class for all transitions in the Petri net system.

    In the autonomous model, transitions run as independent tasks
    waiting on their input places
    """

    def __init__(self):
        self._fire_count = 0
        self._last_fired = None
        self._net: Optional["PetriNet"] = None
        self._qualified_name: Optional[str] = None
        self._interface_path: Optional[str] = None
        self._output_places: Dict[PlaceName, Place] = {}
        self._input_places: Dict[PlaceName, Place] = {}

        self._task: Optional[Task] = None
        self._waiting: bool = False

    def _attach_to_net(self, net: "PetriNet"):
        """Called when transition is attached to a net."""
        self._net = net

        self._input_places = {}
        self._output_places = {}

        output_arcs = self.output_arcs()
        for label, arc in output_arcs.items():
            place = self._net._resolve_place_from_arc(arc)
            self._output_places[label] = place
        input_arcs = self.input_arcs()
        for label, arc in input_arcs.items():
            place = self._net._resolve_place_from_arc(arc)
            self._input_places[label] = place

    def _set_qualified_name(self, qualified_name: str):
        """Set the fully qualified name for this transition."""
        self._qualified_name = qualified_name

    def _set_interface_path(self, interface_path: str):
        """Set the interface path for boundary checking."""
        self._interface_path = interface_path

    @property
    def net(self) -> "PetriNet":
        if not self._net:
            raise ValueError("Transition is not attached to a Petri net.")
        return self._net

    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name for this transition."""
        return self._qualified_name or self.__class__.__name__

    def input_arcs(self) -> Dict[PlaceName, Arc]:
        """Override this method to define input arcs using PlaceName enums."""
        return {}

    def output_arcs(self) -> Dict[PlaceName, Arc]:
        """Override this method to define output arcs using PlaceName enums."""
        return {}

    @property
    def fire_count(self) -> int:
        """Number of times this transition has fired."""
        return self._fire_count

    @property
    def last_fired(self) -> Optional[datetime]:
        """Timestamp of last firing."""
        return self._last_fired

    def guard(self, tokens: Dict[PlaceName, List[Any]]) -> bool:
        """Check if the transition can fire with the given tokens."""
        return True

    def _start_transition_task(self):
        """Start the transition processing task."""
        if self._task is None:
            self.net.add_task_fn(self._transition_loop)

    async def _transition_loop(self):
        """Main transition loop - processes token batches ."""
        try:
            while True:
                # Wait until all input places have _enough_ tokens to meet the arc weight requirement
                self._waiting = True
                for label, arc in self.input_arcs().items():
                    place = self._input_places[label]
                    while not place.token_count >= arc.weight:
                        # Every added token will pulse an event, so we can wait for that OR the shutdown event
                        await place.wait_for_tokens()
                # Generate combinations for each input place
                guard_passed = False
                token_combinations = {}
                for label, arc in self.input_arcs().items():
                    place = self._input_places[label]
                    token_combinations[label] = place.combinations(arc.weight)

                # Generate the Cartesian product of token combinations.
                # Transitions need to check against all possible
                # combinations of tokens in their input places
                token_batch = {}
                all_combinations = product(*token_combinations.values())
                for combo in all_combinations:
                    token_batch = dict(zip(token_combinations.keys(), combo))                        
                    if self.guard(token_batch):
                        guard_passed = True
                        # Remove tokens from input places
                        for label, tokens in token_batch.items():
                            place = self._input_places[label]
                            for token in tokens:
                                place.remove_token(token)
                        break

                # No valid combination found, yield and then wait for new tokens
                if not guard_passed:
                    await sleep(0)
                    continue
                self._waiting = False

                # Fire the transition
                await self._fire_with_tokens(token_batch)

        except CancelledError:
            pass
        finally:
            if self._net and self._task in self._net._running_tasks:
                self._net._running_tasks.discard(self._task)

    async def _fire_with_tokens(self, consumed_tokens: Dict[PlaceName, List[Any]]):
        """Fire the transition with the given consumed tokens."""
        await self.on_before_fire(consumed_tokens)

        # Call user-defined firing logic
        output_tokens = await self.on_fire(consumed_tokens)

        # Produce output tokens
        if output_tokens:
            output_arcs = self.output_arcs()
            for label, tokens in output_tokens.items():
                if label in output_arcs:
                    place = self._output_places[label]
                    arc = output_arcs[label]
                    if not isinstance(tokens, (list, tuple)):
                        tokens = [tokens]
                    for token in tokens:
                        await place.add_token(token)

        self._fire_count += 1
        self._last_fired = datetime.now()

        await self.on_after_fire(consumed_tokens, output_tokens)

    async def on_fire(
        self, consumed: Dict[PlaceName, List[Any]]
    ) -> Optional[Dict[PlaceName, List[Any]]]:
        """
        User-defined firing logic. Override this method.

        Args:
            consumed: Dict mapping PlaceName enums to lists of consumed tokens

        Returns:
            Dict mapping PlaceName enums to lists of tokens to produce,
            or None if no outputs
        """
        # Default: pass through all consumed tokens to all outputs
        output_arcs = self.output_arcs()
        if not output_arcs:
            return None

        result = {}
        for out_label in output_arcs.keys():
            result[out_label] = []
            for tokens in consumed.values():
                result[out_label].extend(tokens)

        return result

    async def on_before_fire(self, consumed: Dict[PlaceName, List[Any]]):
        """Called before transition fires."""
        pass

    async def on_after_fire(
        self,
        consumed: Dict[PlaceName, List[Any]],
        produced: Optional[Dict[PlaceName, List[Any]]],
    ):
        """Called after transition fires successfully."""
        pass


    async def _rollback_tokens(self, consumed_tokens: Dict[PlaceName, List[Any]]):
        """Put consumed tokens back into their places for error recovery."""
        input_arcs = self.input_arcs()

        for label, tokens in consumed_tokens.items():
            if label in input_arcs:
                arc = input_arcs[label]
                place = self.net._resolve_place_from_arc(arc)

                # Put tokens back
                place._tokens.extend(tokens)


    def is_waiting(self) -> bool:
        """Check if this transition is waiting for tokens."""
        return self._waiting

    def __repr__(self):
        return f"{self.qualified_name}(fired={self.fire_count})"

class DeclarativeMeta(type):
    """Metaclass that handles declarative composition."""

    @classmethod
    def __prepare__(mcls, name, bases, **kwargs):
        """
        Return the mapping used as `locals()` while executing the class body.
        We push it onto a context stack so helpers can inject into it.
        """
        ns = _DeclarativeNamespace(mcls)
        _decl_stack_push(ns)
        return ns

    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Build the class from the prepared mapping, then run the discovery you already had.
        We pop the mapping off the context stack afterwards.
        """
        try:
            # IMPORTANT: pass a plain dict into type.__new__
            cls = super().__new__(mcs, name, bases, dict(namespace), **kwargs)

            # Find all nested classes and organize them
            cls._discovered_places = []
            cls._discovered_transitions = []
            cls._discovered_interfaces = []

            # TODO: Make these tolerant of instances (need to back them down to types)
            for attr_name, attr_value in namespace.items():
                if inspect.isclass(attr_value):
                    attr_value._outer_class = cls
                    attr_value._attr_name = attr_name

                    if issubclass(attr_value, Place):
                        cls._discovered_places.append(attr_value)
                    elif issubclass(attr_value, Transition):
                        cls._discovered_transitions.append(attr_value)
                    elif issubclass(attr_value, Interface):
                        cls._discovered_interfaces.append(attr_value)
                elif isinstance(attr_value, list) and all(
                    inspect.isclass(x) and issubclass(x, Place) for x in attr_value
                ):
                    cls._discovered_places.extend(attr_value)
                elif isinstance(attr_value, list) and all(
                    inspect.isclass(x) and issubclass(x, Transition) for x in attr_value
                ):
                    cls._discovered_transitions.extend(attr_value)
                elif isinstance(attr_value, list) and all(
                    inspect.isclass(x) and issubclass(x, Interface) for x in attr_value
                ):
                    cls._discovered_interfaces.extend(attr_value)

            # Inherit from parent classes
            for base in bases:
                if hasattr(base, "_discovered_places"):
                    cls._discovered_places.extend(base._discovered_places)
                if hasattr(base, "_discovered_transitions"):
                    cls._discovered_transitions.extend(base._discovered_transitions)
                if hasattr(base, "_discovered_interfaces"):
                    cls._discovered_interfaces.extend(base._discovered_interfaces)
                    

            return cls

        finally:
            # Always pop the active namespace for this class body
            _decl_stack_pop(namespace)


class DeclarativeABCMeta(DeclarativeMeta, ABCMeta):
    """Combined metaclass that supports both ABC and declarative composition."""

    pass


class Interface(ABC, metaclass=DeclarativeABCMeta):
    """Base class for interface nets using declarative composition."""

    def __init__(self):
        pass

    def get_all_places(self) -> List[Type[Place]]:
        return self._discovered_places

    def get_all_transitions(self) -> List[Type[Transition]]:
        return self._discovered_transitions

    def get_all_interfaces(self) -> List[Type["Interface"]]:
        return self._discovered_interfaces


# Main Petri Net Engine
class PetriNet(metaclass=DeclarativeABCMeta):
    """
    Main autonomous Petri net execution engine using asyncio.

    This creates an autonomous system where transitions run as independent
    tasks waiting on queues, eliminating the need for central coordination.
    """

    def __init__(self, log_fn=print):
        self._places_by_type: Dict[Type[Place], Place] = {}
        self._places_by_qualified_name: Dict[str, Place] = {}
        self._transitions: List[Transition] = []
        self._io_input_places: List[IOInputPlace] = []
        self._io_output_places: List[IOOutputPlace] = []

        self._log_fn = log_fn or (lambda x: None)
        self._typecheck_arcs: bool = False

        # Event handling for coordinated shutdown
        self._shutdown_event = Event()
        self._running_tasks: Set[Task] = set()

        # Tracking for qualified names and interface boundaries
        self._qualified_name_registry: Dict[str, Any] = {}
        self._interface_hierarchy: Dict[str, List[str]] = {}

        # Error handling callback
        self._error_handler: Optional[Callable] = None

        # Cycle counter
        self._cycles: float = 0
        self._cycle_task: Optional[Task] = None
        self._cycle_units: Set[float] = set()

        # Deadlock detection
        self._deadlock_check_task: Optional[Task] = None
        self._deadlock_check_interval: float = 1.0  # Check every second

        self._auto_build()

    def set_error_handler(
        self,
        handler: Callable[[Transition, Exception, Dict[PlaceName, List[Any]]], str],
    ):
        """
        Set error handler for transition failures.

        Handler should return one of: "restart", "kill", or re-raise the exception.
        """
        self._error_handler = handler

    def get_place(self, place_type: Type[Place]) -> Place:
        """Public API: Get a place instance by its class."""
        return self._get_place_by_type(place_type)

    def _auto_build(self):
        """Automatically build the net from discovered components."""
        # First pass: register all components with qualified names
        self._register_components("", self.__class__)

        # Second pass: validate interface boundaries
        self._validate_interface_boundaries()

        # Third pass: instantiate everything
        self._instantiate_components()

    def _register_components(self, prefix: str, cls: Type):
        """Register components with their qualified names."""
        # Register discovered interfaces
        for interface_class in getattr(cls, "_discovered_interfaces", []):
            interface_name = getattr(
                interface_class, "_attr_name", interface_class.__name__
            )
            qualified_name = f"{prefix}.{interface_name}" if prefix else interface_name

            if qualified_name in self._qualified_name_registry:
                raise QualifiedNameConflict(
                    f"Qualified name '{qualified_name}' already exists"
                )

            self._qualified_name_registry[qualified_name] = interface_class
            self._interface_hierarchy[qualified_name] = []

            # Recursively register nested components
            self._register_components(qualified_name, interface_class)

        # Register discovered places
        for place_class in getattr(cls, "_discovered_places", []):
            place_name = getattr(place_class, "_attr_name", place_class.__name__)
            qualified_name = f"{prefix}.{place_name}" if prefix else place_name

            if qualified_name in self._qualified_name_registry:
                raise QualifiedNameConflict(
                    f"Qualified name '{qualified_name}' already exists"
                )

            self._qualified_name_registry[qualified_name] = place_class

            # Track which interface this belongs to
            if prefix:
                self._interface_hierarchy[prefix].append(qualified_name)

        # Register discovered transitions
        for transition_class in getattr(cls, "_discovered_transitions", []):
            transition_name = getattr(
                transition_class, "_attr_name", transition_class.__name__
            )
            qualified_name = (
                f"{prefix}.{transition_name}" if prefix else transition_name
            )

            if qualified_name in self._qualified_name_registry:
                raise QualifiedNameConflict(
                    f"Qualified name '{qualified_name}' already exists"
                )

            self._qualified_name_registry[qualified_name] = transition_class

            # Track which interface this belongs to
            if prefix:
                self._interface_hierarchy[prefix].append(qualified_name)

    def _validate_interface_boundaries(self):
        """Validate that transitions don't cross interface boundaries."""
        pass

    def _validate_transition_boundaries(
        self, transition_qualified_name: str, transition_class: Type[Transition]
    ):
        """Validate that a transition doesn't violate interface boundaries."""
        pass

    def _validate_arc_boundary(
        self,
        place_ref: Union[Type[Place], str],
        transition_interface: str,
        transition_name: str,
        arc_type: str,
        skip_boundary_check: bool = False,
    ):
        """Validate that an arc doesn't cross interface boundaries or reference missing places."""
        pass

    def _is_within_interface_boundary(
        self, place_qualified_name: str, transition_interface: str
    ) -> bool:
        """Allow access to any place that is a direct child or descendant of the transition's interface."""
        return True

    def _instantiate_components(self):
        """Instantiate all registered components."""
        # Instantiate places
        for qualified_name, component in self._qualified_name_registry.items():
            if inspect.isclass(component) and issubclass(component, Place):
                place_instance = component()
                place_instance._set_qualified_name(qualified_name)
                place_instance._attach_to_net(self)

                # Track IO places separately
                if isinstance(place_instance, IOInputPlace):
                    self._io_input_places.append(place_instance)
                elif isinstance(place_instance, IOOutputPlace):
                    self._io_output_places.append(place_instance)

                self._places_by_type[component] = place_instance
                self._places_by_qualified_name[qualified_name] = place_instance

        # Instantiate transitions
        for qualified_name, component in self._qualified_name_registry.items():
            if inspect.isclass(component) and issubclass(component, Transition):
                transition_instance = component()
                transition_instance._set_qualified_name(qualified_name)

                # Find interface path for boundary checking
                interface_path = None
                for interface_name, components in self._interface_hierarchy.items():
                    if qualified_name in components:
                        interface_path = interface_name
                        break

                if interface_path:
                    transition_instance._set_interface_path(interface_path)

                transition_instance._attach_to_net(self)
                self._transitions.append(transition_instance)

    async def log(self, message: str):
        """Log a message."""
        self._log_fn(f"[{datetime.now()}] {message}")

    def _resolve_place_from_arc(self, arc: Arc) -> Place:
        """Resolve a place from an arc, handling type and string references."""
        if isinstance(arc.place, str):
            return self._get_place_by_qualified_name(arc.place)
        else:
            return self._get_place_by_type(arc.place)

    def _get_place_by_type(self, place_type: Type[Place]) -> Place:
        """Get a place by type."""
        if place_type not in self._places_by_type:
            raise KeyError(
                f"Place type {place_type.__name__} not found in {self._places_by_type}"
            )
        return self._places_by_type[place_type]

    def _get_place_by_qualified_name(self, qualified_name: str) -> Place:
        """Get a place by qualified name."""
        if qualified_name not in self._places_by_qualified_name:
            raise KeyError(f"Place {qualified_name} not found")
        return self._places_by_qualified_name[qualified_name]

    async def produce_token(self, place_type: Type[Place], token: Any):
        """Produce a token to the specified place."""
        place = self._get_place_by_type(place_type)
        await place.add_token(token)

    async def _cycle_loop(self):
        """Cycle counter loop."""
        try:
            while True:
                await sleep(0)
                if self._cycle_units:
                    self._cycles += min(self._cycle_units)
                else:
                    self._cycles += 1
        except CancelledError:
            raise

    async def sleep_cycles(self, duration: float):
        """Sleep for a given number of cycles."""
        self._cycle_units.add(duration)
        start = self._cycles
        while self._cycles - start < duration:
            await sleep(0)
        try:
            self._cycle_units.remove(duration)
        except KeyError:
            pass

    async def _deadlock_check_loop(self):
        """Periodically check for deadlocks."""
        try:
            while True:
                await sleep(self._deadlock_check_interval)
                if self._shutdown_event.is_set():
                    break
                if await self.is_deadlocked():
                    await self.log("DEADLOCK DETECTED!")
                    await self.on_deadlock()

        except CancelledError:
            raise
        
    def add_task_fn(self, task_fn) -> asyncio.Task:
        """ Add a guarded task to the event loop and return a handle to it"""
        async def run_guarded(aw):
            try:
                await aw
            except:
                traceback.print_exc()
                sys.exit(1)

        task = create_task(run_guarded(task_fn()))
        self._running_tasks.add(task)
        return task

    async def is_deadlocked(self) -> bool:
        """
        Check if the net is deadlocked.

        A deadlock occurs when:
        1. All transitions are waiting for tokens
        2. No IO input places are active (or they've signaled completion)
        3. There are no tokens in transit
        """
        # Check if all transitions are waiting
        all_waiting = all(t.is_waiting() for t in self._transitions)

        if not all_waiting:
            return False

        # Check if there are any active IO input tasks that might produce tokens
        active_inputs = any(
            place._task and not place._task.done() for place in self._io_input_places
        )

        if active_inputs:
            # Still have active inputs, not deadlocked yet
            return False

        # All transitions waiting, no active inputs
        return True

    async def on_deadlock(self):
        """Called when a deadlock is detected. Override to customize behavior."""
        # Default: raise exception to stop the net
        raise PetriNetDeadlock("Petri net is deadlocked - no transitions can fire")

    async def start(self, typecheck_arcs: bool = False):
        """Start the autonomous Petri net execution."""

        await self.log(f"Starting Net: {self.__class__.__name__}")
        self._shutdown_event.clear()
        self._typecheck_arcs = typecheck_arcs

        # Start our cycle counter
        self._cycles = 0
        self._cycle_task = create_task(self._cycle_loop())
        self._running_tasks.add(self._cycle_task)

        # Start deadlock detector
        self._deadlock_check_task = create_task(self._deadlock_check_loop())
        self._running_tasks.add(self._deadlock_check_task)

        # Start all IO input places
        for io_input_place in self._io_input_places:
            io_input_place._start_input_task()

        # Start all transitions
        for transition in self._transitions:
            transition._start_transition_task()

        await self.log(f"Net: {self.__class__.__name__} tasks started")

    async def stop(self, timeout: Optional[float] = 0):
        """Stop the autonomous Petri net execution."""
        if self._shutdown_event.is_set():
            return

        await self.log(f"Stopping Net: {self.__class__.__name__}")
        self._shutdown_event.set()

        try:
            # Wait for all tasks to complete with timeout
            await wait_for(
                gather(
                    *[task for task in self._running_tasks if not task.done()],
                    return_exceptions=True,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            pass

        self._running_tasks.clear()
        await self.log(f"Net: {self.__class__.__name__} stopped")

    async def run_for(self, duration: float):
        """Run the net for a specified duration."""
        await self.start()
        try:
            await sleep(duration)
        finally:
            await self.stop()

    def get_state_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current net state."""
        return {
            "is_running": not self._shutdown_event.is_set(),
            "places": {
                place.qualified_name: {
                    "token_count": place.token_count,
                    "is_full": place.is_full,
                    "is_empty": place.is_empty,
                }
                for place in self._places_by_type.values()
            },
            "transitions": {
                transition.qualified_name: {
                    "fire_count": transition.fire_count,
                    "last_fired": transition.last_fired,
                    "is_waiting": transition.is_waiting(),
                }
                for transition in self._transitions
            },
            "running_tasks": len(self._running_tasks),
        }

    def generate_mermaid_diagram(self, left_to_right: bool = False) -> str:
        """Generate a Mermaid diagram representing the Petri net structure with nested interface subgraphs."""
        lines = ["graph LR" if left_to_right else "graph TD"]

        # Build a nested tree of interfaces
        def build_interface_tree():
            tree = {}
            # Sort keys so parents come before children
            for interface in sorted(
                self._interface_hierarchy.keys(), key=lambda x: x.count(".")
            ):
                parts = interface.split(".")
                node = tree
                for part in parts:
                    node = node.setdefault(part, {})
                node["_components"] = self._interface_hierarchy[interface]
            return tree

        # Recursively emit subgraphs for all interfaces, nesting children
        def emit_subgraph(node, name="", indent=1, parent_path=""):
            if name:
                interface_name: str = reduce(
                    lambda x, y: f"{x}.{y}" if x else y, [parent_path, name], ""
                )
                safe_interface_name: str = interface_name.replace(".", "_")
                lines.append(
                    "    " * indent
                    + f"subgraph {safe_interface_name} ['{interface_name}']"
                )
            # Emit components and child interfaces
            for key, child in node.items():
                if key == "_components":
                    for comp in child:
                        # If comp is a child interface, skip (will be handled as a node)
                        if comp in self._interface_hierarchy:
                            continue
                        safe_component_name = comp.replace(".", "_")
                        lines.append("    " * (indent + 1) + safe_component_name)
                else:
                    new_parent_path: str = (
                        reduce(
                            lambda x, y: f"{x}.{y}" if x else y,
                            [parent_path, name],
                            "",
                        )
                        or key
                    )
                    emit_subgraph(child, key, indent + 1, new_parent_path)
            if name:
                lines.append("    " * indent + "end")

        # Add all places and transitions as nodes (flat, so arcs can reference them)
        for place in self._places_by_type.values():
            qualified_name = place.qualified_name
            safe_name = qualified_name.replace(".", "_")
            token_count = place.token_count
            if isinstance(place, IOInputPlace):
                lines.append(
                    f'    {safe_name}(("→ {qualified_name}<br/>({token_count} tokens)<br/>[INPUT]"))'
                )
            elif isinstance(place, IOOutputPlace):
                lines.append(
                    f'    {safe_name}(("{qualified_name} →<br/>({token_count} tokens)<br/>[OUTPUT]"))'
                )
            else:
                lines.append(
                    f'    {safe_name}(("{qualified_name}<br/>({token_count} tokens)"))'
                )
        for transition in self._transitions:
            qualified_name = transition.qualified_name
            safe_name = qualified_name.replace(".", "_")
            fire_count = transition.fire_count
            lines.append(
                f'    {safe_name}["{qualified_name}<br/>(fired {fire_count}x)"]'
            )

        # Recursively emit subgraphs starting from the tree root
        tree = build_interface_tree()
        for key, node in tree.items():
            emit_subgraph(node, key, indent=1, parent_path="")

        # Add arcs
        for transition in self._transitions:
            transition_safe_name = transition.qualified_name.replace(".", "_")
            try:
                input_arcs = transition.input_arcs()
                for enum_key, arc in input_arcs.items():
                    place = self._resolve_place_from_arc(arc)
                    place_safe_name = place.qualified_name.replace(".", "_")
                    edge_label = enum_key.name
                    if arc.weight > 1:
                        edge_label += f" ({arc.weight})"
                    lines.append(
                        f'    {place_safe_name} -->|"{edge_label}"| {transition_safe_name}'
                    )
            except Exception as e:
                print(
                    f"Warning: Could not generate input arcs for {transition.qualified_name}: {e}"
                )
            try:
                output_arcs = transition.output_arcs()
                for enum_key, arc in output_arcs.items():
                    place = self._resolve_place_from_arc(arc)
                    place_safe_name = place.qualified_name.replace(".", "_")
                    edge_label = enum_key.name
                    if arc.weight > 1:
                        edge_label += f" ({arc.weight})"
                    lines.append(
                        f'    {transition_safe_name} -->|"{edge_label}"| {place_safe_name}'
                    )
            except Exception as e:
                print(
                    f"Warning: Could not generate output arcs for {transition.qualified_name}: {e}"
                )

        return "\n".join(lines)

    def save_mermaid_diagram(self, filename: str, left_to_right: bool = False):
        """Save the Mermaid diagram to a file."""
        with open(filename, "w") as f:
            f.write(self.generate_mermaid_diagram(left_to_right=left_to_right))

    def print_mermaid_diagram(self, left_to_right: bool = False):
        """Print the Mermaid diagram to console."""
        print(self.generate_mermaid_diagram(left_to_right=left_to_right))

    def __repr__(self):
        return f"PetriNet(places={len(self._places_by_type)}, transitions={len(self._transitions)}, running={not self._shutdown_event.is_set()})"
