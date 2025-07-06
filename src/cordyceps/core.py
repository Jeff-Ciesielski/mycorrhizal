#!/usr/bin/env python3
"""
Cordyceps Async Core - Event-driven Petri net library

This is the foundation for an async-first Petri net implementation.
"""

import asyncio
import inspect
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
import weakref
from collections import defaultdict


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


# Core Token System
class Token(ABC):
    """
    Base class for all tokens in the Petri net system.
    
    Tokens are the data carriers that flow through the system between places.
    """
    
    def __init__(self, data: Any = None):
        self.data = data
        self.created_at = datetime.now()
        self.id = id(self)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, data={self.data})"
    
    async def on_created(self):
        """Called when token is first created."""
        pass
    
    async def on_consumed(self):
        """Called when token is consumed by a transition."""
        pass
    
    async def on_produced(self):
        """Called when token is produced into a place."""
        pass


class Place(ABC):
    """
    Base class for all places in the Petri net system.
    
    Places hold tokens and notify the net when tokens are added/removed.
    In the async model, places are event sources that trigger net updates.
    """
    
    MAX_CAPACITY: Optional[int] = None
    ACCEPTED_TOKEN_TYPES: Optional[Set[Type[Token]]] = None
    
    def __init__(self):
        self._tokens: List[Token] = []
        self._total_tokens_processed = 0
        self._net: Optional['PetriNet'] = None
        self._token_added_callbacks: List[Callable] = []
        self._qualified_name: Optional[str] = None
        
    def _attach_to_net(self, net: 'PetriNet'):
        """Called when place is attached to a net."""
        self._net = net
        
    def _set_qualified_name(self, qualified_name: str):
        """Set the fully qualified name for this place."""
        self._qualified_name = qualified_name
        
    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name for this place."""
        return self._qualified_name or self.__class__.__name__
        
    @property
    def token_count(self) -> int:
        """Returns the current number of tokens in this place."""
        return len(self._tokens)
    
    @property
    def tokens(self) -> List[Token]:
        """Returns a copy of the current tokens."""
        return self._tokens.copy()
    
    @property
    def is_empty(self) -> bool:
        """Returns True if the place has no tokens."""
        return len(self._tokens) == 0
    
    @property
    def is_full(self) -> bool:
        """Returns True if the place is at maximum capacity."""
        return self.MAX_CAPACITY is not None and len(self._tokens) >= self.MAX_CAPACITY
    
    def can_accept_token(self, token: Token) -> bool:
        """Check if this place can accept the given token."""
        if self.is_full:
            return False
            
        if self.ACCEPTED_TOKEN_TYPES is not None:
            return type(token) in self.ACCEPTED_TOKEN_TYPES
            
        return True
    
    async def add_token(self, token: Token):
        # Debug print removed
        """
        Add a token to this place and notify the net.
        
        This is the core event that drives the Petri net execution.
        """
        if not self.can_accept_token(token):
            if self.is_full:
                raise InsufficientTokens(f"Place {self.qualified_name} is at capacity")
            else:
                raise InvalidTokenType(f"Place {self.qualified_name} cannot accept token type {type(token)}")
        
        self._tokens.append(token)
        self._total_tokens_processed += 1
        
        await token.on_produced()
        await self.on_token_added(token)
        
        # Notify the net that a token was added - this triggers transition checking
        if self._net:
            await self._net._on_token_added(self, token)
    
    async def remove_token(self, token_type: Type[Token] = None) -> Token:
        # Debug print removed
        """Remove and return a token from this place."""
        if self.is_empty:
            raise InsufficientTokens(f"No tokens available in place {self.qualified_name}")
        
        if token_type is None:
            token = self._tokens.pop(0)  # FIFO by default
        else:
            # Find first token of specified type
            for i, token in enumerate(self._tokens):
                if isinstance(token, token_type):
                    token = self._tokens.pop(i)
                    break
            else:
                raise InsufficientTokens(f"No tokens of type {token_type.__name__} in place {self.qualified_name}")
        
        await token.on_consumed()
        await self.on_token_removed(token)
        
        # Notify the net that a token was removed
        if self._net:
            await self._net._on_token_removed(self, token)
        
        return token
    
    def peek_token(self, token_type: Type[Token] = None) -> Optional[Token]:
        """Return a token without removing it from the place."""
        if self.is_empty:
            return None
            
        if token_type is None:
            return self._tokens[0]
        else:
            for token in self._tokens:
                if isinstance(token, token_type):
                    return token
            return None
    
    def count_tokens(self, token_type: Type[Token] = None) -> int:
        """Count tokens of a specific type."""
        if token_type is None:
            return len(self._tokens)
        else:
            return sum(1 for token in self._tokens if isinstance(token, token_type))
    
    async def on_token_added(self, token: Token):
        """Called when a token is added to this place."""
        pass
    
    async def on_token_removed(self, token: Token):
        """Called when a token is removed from this place."""
        pass
    
    def __repr__(self):
        return f"{self.qualified_name}(tokens={self.token_count})"


class IOOutputPlace(Place):
    """
    Base class for output IO places that send tokens to external systems.
    
    IOOutputPlaces consume tokens and send them to external systems.
    If the external operation fails, they can produce error tokens.
    """
    
    async def on_token_added(self, token: Token):
        """
        Handle token addition - this is where output IO happens.
        
        This calls on_token() and handles the response:
        - If on_token() returns None, the token is consumed (success)
        - If on_token() returns an error token, the original is replaced
        """
        try:
            # Call the user-defined handler
            error_token = await self.on_token(token)
            
            if error_token is None:
                # Success - remove the token that was just added
                self._tokens.remove(token)
                await self.on_io_success(token)
            else:
                # Failed - replace the token with an error token
                self._tokens.remove(token)
                self._tokens.append(error_token)
                await self.on_io_error(token, error_token)
                
        except Exception as e:
            # Handle unexpected errors
            self.log(f"Unexpected error in IO output place {self.qualified_name}: {e}")
            await self.on_io_error(token, None)
    
    async def on_token(self, token: Token) -> Optional[Token]:
        """
        Handle outgoing token. Override this for output places.
        
        Returns:
            None if successful (token is consumed)
            Error token if failed (replaces original token)
        """
        # Default implementation just logs and succeeds
        self.log(f"IO Output: {token}")
        return None
    
    async def on_io_success(self, token: Token):
        """Called when IO operation succeeds."""
        pass
    
    async def on_io_error(self, token: Token, error_token: Optional[Token]):
        """Called when IO operation fails."""
        pass
    
    def log(self, message: str):
        """Log a message."""
        if self._net:
            self._net.log(message)
        else:
            print(message)


class IOInputPlace(Place):
    """
    Base class for input IO places that receive tokens from external systems.
    
    IOInputPlaces generate tokens from external events/sources.
    They run background tasks that produce tokens into the place.
    """
    
    def __init__(self):
        super().__init__()
        self._input_task = None
        self._shutdown_event = None  # Will be created when needed
    
    def _ensure_shutdown_event(self):
        """Create shutdown event if it doesn't exist."""
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
    
    async def start_input(self):
        """Start the input task if not already running."""
        self._ensure_shutdown_event()
        
        if self._input_task is None or self._input_task.done():
            self._input_task = asyncio.create_task(self._input_loop())
            if self._net:
                self._net._running_tasks.add(self._input_task)
                self._input_task.add_done_callback(self._net._running_tasks.discard)
    
    async def stop_input(self):
        """Stop the input task gracefully."""
        self._ensure_shutdown_event()
        self._shutdown_event.set()
        
        if self._input_task and not self._input_task.done():
            try:
                await asyncio.wait_for(self._input_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._input_task.cancel()
                await asyncio.gather(self._input_task, return_exceptions=True)
    
    async def _input_loop(self):
        """Main input loop - calls on_input() until shutdown."""
        self._ensure_shutdown_event()
        
        try:
            await self.on_input_start()
            
            while not self._shutdown_event.is_set():
                try:
                    # Call user-defined input handler
                    await self.on_input()
                    
                    # Small delay to prevent busy waiting
                    await asyncio.sleep(0.01)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.log(f"Error in input loop for {self.qualified_name}: {e}")
                    await asyncio.sleep(0.1)  # Back off on error
            
            await self.on_input_stop()
            
        except asyncio.CancelledError:
            await self.on_input_stop()
            raise
    
    async def on_input_start(self):
        """Called when input starts."""
        pass
    
    async def on_input(self):
        """
        Main input handler - override this to generate tokens.
        
        This method is called repeatedly until shutdown.
        Use self.produce_token() to add tokens to this place.
        """
        # Default implementation waits for shutdown
        self._ensure_shutdown_event()
        await self._shutdown_event.wait()
    
    async def on_input_stop(self):
        """Called when input stops."""
        pass
    
    async def produce_token(self, token: Token):
        """Produce a token into this place."""
        await self.add_token(token)
    
    def log(self, message: str):
        """Log a message."""
        if self._net:
            self._net.log(message)
        else:
            print(message)


@dataclass
class Arc:
    """Represents a connection between a place and transition with metadata."""
    place: Union[Type[Place], str]  # Allow string references for deferred resolution
    weight: int = 1
    token_type: Optional[Type[Token]] = None
    label: Optional[str] = None
    
    def __post_init__(self):
        if self.label is None:
            if isinstance(self.place, str):
                self.label = self.place.split('.')[-1].lower()
            elif self.place is not None:
                self.label = self.place.__name__.lower()
            else:
                self.label = "unnamed_arc"


# Core Transition System
class Transition(ABC):
    """
    Base class for all transitions in the Petri net system.
    
    In the async model, transitions are checked for firing whenever
    tokens are added to their input places.
    """
    PRIORITY: int = 0  # Higher values fire first

    def __init__(self):
        self._fire_count = 0
        self._last_fired = None
        self._net: Optional['PetriNet'] = None
        self._qualified_name: Optional[str] = None
        self._interface_path: Optional[str] = None

    def _attach_to_net(self, net: 'PetriNet'):
        """Called when transition is attached to a net."""
        self._net = net
        
    def _set_qualified_name(self, qualified_name: str):
        """Set the fully qualified name for this transition."""
        self._qualified_name = qualified_name
        
    def _set_interface_path(self, interface_path: str):
        """Set the interface path for boundary checking."""
        self._interface_path = interface_path
        
    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name for this transition."""
        return self._qualified_name or self.__class__.__name__

    def input_arcs(self):
        """Override this method to define input arcs."""
        return {}

    def output_arcs(self):
        """Override this method to define output arcs."""
        return {}

    @property
    def fire_count(self) -> int:
        """Number of times this transition has fired."""
        return self._fire_count

    @property
    def last_fired(self) -> Optional[datetime]:
        """Timestamp of last firing."""
        return self._last_fired


    def __init__(self):
        self._fire_count = 0
        self._last_fired = None
        self._net: Optional['PetriNet'] = None
        self._qualified_name: Optional[str] = None
        self._interface_path: Optional[str] = None
        self._to_consume = {}
        self._to_produce = {}

    def consume(self, label, token):
        """Mark a token for consumption from a given input arc label."""
        if label not in self._to_consume:
            self._to_consume[label] = []
        self._to_consume[label].append(token)

    def produce(self, label, token):
        """Mark a token to be produced to a given output arc label."""
        if label not in self._to_produce:
            self._to_produce[label] = []
        self._to_produce[label].append(token)

    async def can_fire(self):
        """Check if this transition can fire. Returns True if fireable, else False."""
        self._to_consume = {}
        input_arcs = self._normalize_arcs(self.input_arcs())
        pending = {}
        for label, arc in input_arcs.items():
            place = self._net._resolve_place_from_arc(arc)
            tokens = [t for t in place.tokens if arc.token_type is None or isinstance(t, arc.token_type)]
            pending[label] = tokens
        result = await self.guard(pending)
        # Only fire if guard returns True and at least one token is marked for consumption (or no input arcs)
        if result is True and (self._to_consume or not input_arcs):
            return True
        return False

    async def fire(self):
        # Debug print removed
        """
        Fire this transition. The guard determines which tokens to consume from each input arc.
        """
        if not await self.can_fire():
            return False

        input_arcs = self._normalize_arcs(self.input_arcs())
        consumed_tokens = {}
        # Remove only the tokens marked for consumption
        for label, arc in input_arcs.items():
            place = self._net._resolve_place_from_arc(arc)
            tokens = []
            for token in self._to_consume.get(label, []):
                try:
                    idx = place._tokens.index(token)
                except ValueError:
                    raise InsufficientTokens(f"Token {token} not found in place {place.qualified_name}")
                token = place._tokens.pop(idx)
                await token.on_consumed()
                await place.on_token_removed(token)
                if place._net:
                    await place._net._on_token_removed(place, token)
                tokens.append(token)
            consumed_tokens[label] = tokens

        self._to_produce = {}
        await self.on_before_fire()
        result = await self.on_fire(consumed_tokens)

        # Actually produce tokens
        output_arcs = self._normalize_arcs(self.output_arcs())
        if result is not None:
            for label, tokens in result.items():
                if label in output_arcs:
                    arc = output_arcs[label]
                    place = self._net._resolve_place_from_arc(arc)
                    for token in tokens:
                        await place.add_token(token)
        for label, tokens in self._to_produce.items():
            if label in output_arcs:
                arc = output_arcs[label]
                place = self._net._resolve_place_from_arc(arc)
                for token in tokens:
                    await place.add_token(token)

        self._fire_count += 1
        self._last_fired = datetime.now()
        await self.on_after_fire()
        return True

    async def guard(self, pending):
        """
        User should override this. Inspect `pending` (dict of arc label to tokens),
        call self.consume(label, token) for any tokens to consume, and return True if should fire.
        """
        # Default: consume first token from each arc if available
        for label, tokens in pending.items():
            if tokens:
                self.consume(label, tokens[0])
            else:
                return False
        return True

    async def on_fire(self, consumed: Dict[str, List[Token]]) -> Optional[Dict[str, List[Token]]]:
        """
        User should override this. Use self.produce(label, token) to produce output tokens.
        """
        # Default: produce consumed tokens to all output arcs
        for label, tokens in consumed.items():
            for out_label in self._normalize_arcs(self.output_arcs()).keys():
                for token in tokens:
                    self.produce(out_label, token)
    
    async def on_before_fire(self):
        """Called before transition fires."""
        pass
    
    async def on_after_fire(self):
        """Called after transition fires."""
        pass

    def _normalize_arcs(self, arcs):
        """Normalize arcs to dict format."""
        if isinstance(arcs, list):
            return {str(i): arc for i, arc in enumerate(arcs)}
        return arcs

    def __repr__(self):
        return f"{self.qualified_name}(fired={self.fire_count})"


# Declarative Composition System
class DeclarativeMeta(type):
    """Metaclass that handles declarative composition."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Find all nested classes and organize them
        cls._discovered_places = []
        cls._discovered_transitions = []
        cls._discovered_interfaces = []
        
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
        
        # Inherit from parent classes
        for base in bases:
            if hasattr(base, '_discovered_places'):
                cls._discovered_places.extend(base._discovered_places)
            if hasattr(base, '_discovered_transitions'):
                cls._discovered_transitions.extend(base._discovered_transitions)
            if hasattr(base, '_discovered_interfaces'):
                cls._discovered_interfaces.extend(base._discovered_interfaces)
        
        return cls


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
    
    def get_all_interfaces(self) -> List[Type['Interface']]:
        return self._discovered_interfaces


# Main Petri Net Engine
class PetriNet(metaclass=DeclarativeABCMeta):
    def get_place(self, place_type: type) -> Place:
        """Public API: Get a place instance by its class (not string)."""
        return self._get_place_by_type(place_type)
    """
    Main async Petri net execution engine.
    
    This is event-driven: transitions are checked whenever tokens are added to places.
    """
    
    def __init__(self, log_fn=print):
        self._places_by_type: Dict[Type[Place], Place] = {}
        self._places_by_qualified_name: Dict[str, Place] = {}
        self._transitions: List[Transition] = []
        self._io_input_places: List[IOInputPlace] = []
        self._io_output_places: List[IOOutputPlace] = []
        # Place -> Transition (input) mapping for efficient firing
        self._place_to_transition: Dict[Place, Transition] = {}

        self._log_fn = log_fn or (lambda x: None)

        self._is_running = False
        self._is_finished = False
        self._firing_log = []

        # Event handling
        self._transition_check_queue = asyncio.Queue()
        self._running_tasks = set()

        # Tracking for qualified names and interface boundaries
        self._qualified_name_registry: Dict[str, Any] = {}
        self._interface_hierarchy: Dict[str, List[str]] = {}

        self._auto_build()
    
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
        for interface_class in getattr(cls, '_discovered_interfaces', []):
            interface_name = getattr(interface_class, '_attr_name', interface_class.__name__)
            qualified_name = f"{prefix}.{interface_name}" if prefix else interface_name
            
            if qualified_name in self._qualified_name_registry:
                raise QualifiedNameConflict(f"Qualified name '{qualified_name}' already exists")
            
            self._qualified_name_registry[qualified_name] = interface_class
            self._interface_hierarchy[qualified_name] = []
            
            # Recursively register nested components
            self._register_components(qualified_name, interface_class)
        
        # Register discovered places
        for place_class in getattr(cls, '_discovered_places', []):
            place_name = getattr(place_class, '_attr_name', place_class.__name__)
            qualified_name = f"{prefix}.{place_name}" if prefix else place_name
            
            if qualified_name in self._qualified_name_registry:
                raise QualifiedNameConflict(f"Qualified name '{qualified_name}' already exists")
            
            self._qualified_name_registry[qualified_name] = place_class
            
            # Track which interface this belongs to
            if prefix:
                self._interface_hierarchy[prefix].append(qualified_name)
        
        # Register discovered transitions
        for transition_class in getattr(cls, '_discovered_transitions', []):
            transition_name = getattr(transition_class, '_attr_name', transition_class.__name__)
            qualified_name = f"{prefix}.{transition_name}" if prefix else transition_name
            
            if qualified_name in self._qualified_name_registry:
                raise QualifiedNameConflict(f"Qualified name '{qualified_name}' already exists")
            
            self._qualified_name_registry[qualified_name] = transition_class
            
            # Track which interface this belongs to
            if prefix:
                self._interface_hierarchy[prefix].append(qualified_name)
    
    def _validate_interface_boundaries(self):
        """Validate that transitions don't cross interface boundaries."""
        self.log("Starting interface boundary validation...")
        
        for qualified_name, component in self._qualified_name_registry.items():
            if inspect.isclass(component) and issubclass(component, Transition):
                self.log(f"Validating transition: {qualified_name}")
                self._validate_transition_boundaries(qualified_name, component)
    
    def _validate_transition_boundaries(self, transition_qualified_name: str, transition_class: Type[Transition]):
        """Validate that a transition doesn't violate interface boundaries."""
        # Find the interface this transition belongs to
        transition_interface = None
        for interface_name, components in self._interface_hierarchy.items():
            if transition_qualified_name in components:
                transition_interface = interface_name
                break
        
        # Create a temporary instance to check arc definitions
        temp_transition = transition_class()

        # Always validate input arcs, and raise InterfaceBoundaryViolation for any missing place
        try:
            input_arcs = temp_transition._normalize_arcs(temp_transition.input_arcs())
        except Exception as e:
            # If arc normalization fails, treat as missing place reference
            raise InterfaceBoundaryViolation(
                f"Transition {transition_qualified_name} has invalid input arcs: {e}"
            )
        for label, arc in input_arcs.items():
            self.log(f"Checking input arc {label}: {arc.place}")
            # For top-level transitions, only check existence, not boundary
            self._validate_arc_boundary(arc.place, transition_interface, transition_qualified_name, "input", skip_boundary_check=(not transition_interface))

        # Always validate output arcs, and raise InterfaceBoundaryViolation for any missing place
        try:
            output_arcs = temp_transition._normalize_arcs(temp_transition.output_arcs())
        except Exception as e:
            # If arc normalization fails, treat as missing place reference
            raise InterfaceBoundaryViolation(
                f"Transition {transition_qualified_name} has invalid output arcs: {e}"
            )
        for label, arc in output_arcs.items():
            self.log(f"Checking output arc {label}: {arc.place}")
            self._validate_arc_boundary(arc.place, transition_interface, transition_qualified_name, "output", skip_boundary_check=(not transition_interface))
    
    def _validate_arc_boundary(self, place_ref: Union[Type[Place], str], transition_interface: str, transition_name: str, arc_type: str, skip_boundary_check: bool = False):
        """Validate that an arc doesn't cross interface boundaries or reference missing places."""
        # Always resolve the qualified name and ensure the place exists
        if isinstance(place_ref, str):
            place_qualified_name = place_ref
            place_type = self._qualified_name_registry.get(place_qualified_name)
        else:
            # Find the qualified name of this place type
            place_qualified_name = None
            place_type = None
            for qualified_name, component in self._qualified_name_registry.items():
                if component == place_ref:
                    place_qualified_name = qualified_name
                    place_type = component
                    break
        if not place_qualified_name or not place_type:
            raise MissingPlaceError(
                f"Transition {transition_name} references unknown place '{getattr(place_ref, '__name__', str(place_ref))}' in {arc_type} arc"
            )

        if skip_boundary_check:
            # Only check existence, not interface boundary
            return

        self.log(f"Checking boundary: place {place_qualified_name} vs transition interface {transition_interface}")

        # Check if the place is within the same interface or a child interface
        if not self._is_within_interface_boundary(place_qualified_name, transition_interface):
            raise InterfaceBoundaryViolation(
                f"Transition {transition_name} violates interface boundary: "
                f"cannot access {place_qualified_name} from interface {transition_interface}"
            )
    
    def _is_within_interface_boundary(self, place_qualified_name: str, transition_interface: str) -> bool:
        """Allow access to any place that is a direct child or any descendant (nested) of the transition's interface."""
        self.log(f"Boundary check: {place_qualified_name} within {transition_interface}?")
        if place_qualified_name == transition_interface:
            # Should not happen (places are not interfaces), but for completeness
            return False
        # Direct child
        direct_children = set(self._interface_hierarchy.get(transition_interface, []))
        if place_qualified_name in direct_children:
            self.log("Place is a direct child of the interface (allowed)")
            return True
        # Any descendant (nested deeper within the interface)
        if place_qualified_name.startswith(transition_interface + "."):
            self.log("Place is a descendant (nested) within the interface (allowed)")
            return True
        self.log("Place violates interface boundary (not a child or descendant)")
        return False
    
    def _instantiate_components(self):
        """Instantiate all registered components and build place->transition mapping."""
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
                self.log(f"Added place: {qualified_name}")

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
                self.log(f"Added transition: {qualified_name}")

        # Build place->transition mapping (enforce only one transition per input place)
        for transition in self._transitions:
            input_arcs = transition._normalize_arcs(transition.input_arcs())
            for label, arc in input_arcs.items():
                place = self._resolve_place_from_arc(arc)
                if place in self._place_to_transition:
                    raise ArcValidationError(
                        f"Place '{place.qualified_name}' is used as input for more than one transition: "
                        f"{self._place_to_transition[place].qualified_name} and {transition.qualified_name}"
                    )
                self._place_to_transition[place] = transition
    
    def log(self, message: str):
        """Log a message."""
        self._log_fn(f"[{datetime.now()}] {message}")
    
    def _resolve_place_from_arc(self, arc: Arc) -> Place:
        """Resolve a place from an arc, handling both type and string references."""
        if isinstance(arc.place, str):
            return self._get_place_by_qualified_name(arc.place)
        else:
            return self._get_place_by_type(arc.place)
    
    def _get_place_by_type(self, place_type: Type[Place]) -> Place:
        """Get a place by type."""
        if place_type not in self._places_by_type:
            raise KeyError(f"Place type {place_type.__name__} not found")
        return self._places_by_type[place_type]
    
    def _get_place_by_qualified_name(self, qualified_name: str) -> Place:
        """Get a place by qualified name."""
        if qualified_name not in self._places_by_qualified_name:
            raise KeyError(f"Place {qualified_name} not found")
        return self._places_by_qualified_name[qualified_name]
    
    async def produce_token(self, place_type: Type[Place], token: Token):
        """Produce a token to the specified place."""
        place = self._get_place_by_type(place_type)
        await place.add_token(token)
    
    async def _on_token_added(self, place: Place, token: Token):
        """Called when a token is added to any place - triggers transition checking."""
        self.log(f"Token added to {place.qualified_name}: {token}")
        # Only queue the transition that uses this place as input, if any
        transition = self._place_to_transition.get(place)
        if transition is not None:
            await self._transition_check_queue.put(transition)
    
    async def _on_token_removed(self, place: Place, token: Token):
        """Called when a token is removed from any place."""
        self.log(f"Token removed from {place.qualified_name}: {token}")
    
    async def _check_and_fire_transitions(self, transitions):
        """Check the given transitions and fire those that can fire."""
        # Sort transitions by priority
        sorted_transitions = sorted(transitions, key=lambda t: t.PRIORITY, reverse=True)
        for transition in sorted_transitions:
            try:
                if await transition.can_fire():
                    fired = await transition.fire()
                    if fired:
                        self.log(f"Fired transition: {transition.qualified_name}")
                        self._firing_log.append((datetime.now(), transition.qualified_name))
            except Exception as e:
                self.log(f"Error in transition {transition.qualified_name}: {e}")
    
    async def _event_loop(self):
        """Main event loop that processes token additions and fires transitions."""
        while self._is_running:
            try:
                # Wait for events
                transition = await asyncio.wait_for(self._transition_check_queue.get(), timeout=0.1)
                # Process the event
                await self._check_and_fire_transitions([transition])
                # Check termination conditions
                if self._should_terminate():
                    self._is_finished = True
                    break
            except asyncio.TimeoutError:
                # No events - check if we should terminate
                if self._should_terminate():
                    self._is_finished = True
                    break
                continue
            except Exception as e:
                self.log(f"Error in event loop: {e}")
                break
        self.log("Event loop finished")
    
    def _should_terminate(self) -> bool:
        """Override this to implement custom termination logic."""
        return False
    
    async def start(self):
        """Start the Petri net execution."""
        if self._is_running:
            return
        
        self._is_running = True
        self.log("Starting Petri net")
        
        # Start all IO input places
        for io_input_place in self._io_input_places:
            await io_input_place.start_input()
        
        # Start main event loop
        try:
            await self._event_loop()
        except Exception as e:
            self.log(f"Error in main loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the Petri net execution."""
        if not self._is_running:
            return
        
        self._is_running = False
        self.log("Stopping Petri net")
        
        # Stop all IO input places
        for io_input_place in self._io_input_places:
            await io_input_place.stop_input()
        
        # Cancel all running tasks
        for task in self._running_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        
        self.log("Petri net stopped")
    
    async def run_until_complete(self, max_iterations: int = 1000):
        """
        Run the net until no more transitions can fire or max iterations reached.
        
        This is useful for batch processing or testing.
        """
        self.log("Starting run_until_complete")
        
        iteration = 0
        while iteration < max_iterations:
            # Check if any transitions can fire
            any_fired = False
            sorted_transitions = sorted(self._transitions, key=lambda t: t.PRIORITY, reverse=True)
            
            for transition in sorted_transitions:
                try:
                    if await transition.can_fire():
                        fired = await transition.fire()
                        if fired:
                            self.log(f"Fired transition: {transition.qualified_name}")
                            self._firing_log.append((datetime.now(), transition.qualified_name))
                            any_fired = True
                except Exception as e:
                    self.log(f"Error in transition {transition.qualified_name}: {e}")
            
            if not any_fired:
                self.log("No more transitions can fire - net is stable")
                break
            
            iteration += 1
            
            # Check termination conditions
            if self._should_terminate():
                self._is_finished = True
                break
        
        if iteration >= max_iterations:
            self.log(f"Stopped after {max_iterations} iterations")
        
        self.log("run_until_complete finished")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current net state."""
        return {
            'is_running': self._is_running,
            'is_finished': self._is_finished,
            'places': {
                place.qualified_name: {
                    'token_count': place.token_count,
                    'tokens': [repr(token) for token in place.tokens]
                }
                for place in self._places_by_type.values()
            },
            'transitions': {
                transition.qualified_name: {
                    'fire_count': transition.fire_count,
                    'last_fired': transition.last_fired
                }
                for transition in self._transitions
            },
            'firing_log': [(str(time), name) for time, name in self._firing_log]
        }
    
    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram representing the Petri net structure."""
        lines = ["graph TD"]

        # Add places with their qualified names
        for place in self._places_by_type.values():
            qualified_name = place.qualified_name
            # Use safe node IDs (replace dots with underscores)
            safe_name = qualified_name.replace(".", "_")
            token_count = place.token_count
            
            if isinstance(place, IOInputPlace):
                lines.append(f'    {safe_name}(("→ {qualified_name}<br/>({token_count} tokens)<br/>[INPUT]"))')
            elif isinstance(place, IOOutputPlace):
                lines.append(f'    {safe_name}(("{qualified_name} →<br/>({token_count} tokens)<br/>[OUTPUT]"))')
            else:
                lines.append(f'    {safe_name}(("{qualified_name}<br/>({token_count} tokens)"))')

        # Add transitions with their qualified names
        for transition in self._transitions:
            qualified_name = transition.qualified_name
            safe_name = qualified_name.replace(".", "_")
            fire_count = transition.fire_count
            lines.append(f'    {safe_name}["{qualified_name}<br/>(fired {fire_count}x)"]')

        # Group nodes in subgraphs for interfaces
        for interface_name, components in self._interface_hierarchy.items():
            if components:
                safe_interface_name = interface_name.replace(".", "_")
                lines.append(f'    subgraph {safe_interface_name} ["{interface_name}"]')
                for component_name in components:
                    safe_component_name = component_name.replace(".", "_")
                    lines.append(f'        {safe_component_name}')
                lines.append('    end')

        # Add arcs
        for transition in self._transitions:
            transition_safe_name = transition.qualified_name.replace(".", "_")

            # Input arcs
            try:
                input_arcs = transition._normalize_arcs(transition.input_arcs())
                for label, arc in input_arcs.items():
                    place = self._resolve_place_from_arc(arc)
                    place_safe_name = place.qualified_name.replace(".", "_")
                    if arc.weight > 1:
                        lines.append(f'    {place_safe_name} -->|"{arc.weight}"| {transition_safe_name}')
                    else:
                        lines.append(f'    {place_safe_name} --> {transition_safe_name}')
            except Exception as e:
                self.log(f"Warning: Could not generate input arcs for {transition.qualified_name}: {e}")

            # Output arcs
            try:
                output_arcs = transition._normalize_arcs(transition.output_arcs())
                for label, arc in output_arcs.items():
                    place = self._resolve_place_from_arc(arc)
                    place_safe_name = place.qualified_name.replace(".", "_")
                    if arc.weight > 1:
                        lines.append(f'    {transition_safe_name} -->|"{arc.weight}"| {place_safe_name}')
                    else:
                        lines.append(f'    {transition_safe_name} --> {place_safe_name}')
            except Exception as e:
                self.log(f"Warning: Could not generate output arcs for {transition.qualified_name}: {e}")

        return "\n".join(lines)
    
    def save_mermaid_diagram(self, filename: str):
        """Save the Mermaid diagram to a file."""
        with open(filename, 'w') as f:
            f.write(self.generate_mermaid_diagram())
    
    def __repr__(self):
        return f"PetriNet(places={len(self._places_by_type)}, transitions={len(self._transitions)})"


# Utility Functions
def create_simple_token(data: Any = None) -> Token:
    """Create a simple token with the given data."""
    class SimpleToken(Token):
        pass
    return SimpleToken(data)