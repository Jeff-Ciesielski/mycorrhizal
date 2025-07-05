#!/usr/bin/env python3
"""
Cordyceps: A type-safe, event-driven Petri net library for concurrent systems.

Fixed version addressing metaclass conflict between ABC and DeclarativeMeta.
"""

import collections
import inspect
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from queue import PriorityQueue, Empty
from typing import Any, Dict, List, Optional, Set, Type, Union
from threading import Timer
import weakref


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


class NonBlockingStalled(Exception):
    """Raised when a non-blocking Petri net stalls."""
    pass


class ArcValidationError(Exception):
    """Raised when arc definitions are invalid."""
    pass


# Core Token System
class Token(ABC):
    """
    Base class for all tokens in the Petri net system.
    
    Tokens are the data carriers that flow through the system between places.
    They can carry arbitrary data and have optional lifecycle methods.
    """
    
    def __init__(self, data: Any = None):
        self.data = data
        self.created_at = datetime.now()
        self.id = id(self)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.id}, data={self.data})"
    
    def on_created(self):
        """Called when token is first created."""
        pass
    
    def on_consumed(self):
        """Called when token is consumed by a transition."""
        pass
    
    def on_produced(self):
        """Called when token is produced into a place."""
        pass


class Place(ABC):
    """
    Base class for all places in the Petri net system.
    
    Places hold tokens and can optionally validate token types and implement
    capacity limits. They act as queues, resource pools, or message buffers.
    
    Places can have external callbacks attached for IO processing when tokens
    are added (useful for output places that need to trigger external actions).
    """
    
    MAX_CAPACITY: Optional[int] = None
    ACCEPTED_TOKEN_TYPES: Optional[Set[Type[Token]]] = None
    
    def __init__(self):
        self._tokens: List[Token] = []
        self._total_tokens_processed = 0
        self._on_token_added_callbacks = []
        
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
        """
        Determines if this place can accept the given token.
        
        Checks both capacity constraints and token type validation.
        """
        if self.is_full:
            return False
            
        if self.ACCEPTED_TOKEN_TYPES is not None:
            return type(token) in self.ACCEPTED_TOKEN_TYPES
            
        return True
    
    def add_token_callback(self, callback):
        """
        Register a callback to be called when tokens are added to this place.
        
        The callback will be called with (token, place) as arguments.
        Useful for implementing IO boundaries where external systems need
        to be notified when tokens arrive at output places.
        
        Args:
            callback: Function that takes (token, place) arguments
        """
        self._on_token_added_callbacks.append(callback)
    
    def remove_token_callback(self, callback):
        """
        Remove a registered callback.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self._on_token_added_callbacks:
            self._on_token_added_callbacks.remove(callback)
    
    def add_token(self, token: Token):
        """
        Adds a token to this place.
        
        Args:
            token: The token to add
            
        Raises:
            InvalidTokenType: If token type is not accepted
            Exception: If place is at capacity
        """
        if not self.can_accept_token(token):
            if self.is_full:
                raise Exception(f"Place {self.__class__.__name__} is at capacity")
            else:
                raise InvalidTokenType(f"Place {self.__class__.__name__} cannot accept token type {type(token)}")
        
        self._tokens.append(token)
        self._total_tokens_processed += 1
        token.on_produced()
        self.on_token_added(token)
        
        # Call external callbacks after token is successfully added
        for callback in self._on_token_added_callbacks:
            try:
                callback(token, self)
            except Exception as e:
                # Log but don't let external callbacks break the net
                print(f"Warning: External callback failed for {self.__class__.__name__}: {e}")
    
    def remove_token(self, token_type: Type[Token] = None) -> Token:
        """
        Removes and returns a token from this place.
        
        Args:
            token_type: Optional specific token type to remove
            
        Returns:
            The removed token
            
        Raises:
            InsufficientTokens: If no matching token is available
        """
        if self.is_empty:
            raise InsufficientTokens(f"No tokens available in place {self.__class__.__name__}")
        
        if token_type is None:
            token = self._tokens.pop(0)  # FIFO by default
        else:
            # Find first token of specified type
            for i, token in enumerate(self._tokens):
                if isinstance(token, token_type):
                    token = self._tokens.pop(i)
                    break
            else:
                raise InsufficientTokens(f"No tokens of type {token_type.__name__} in place {self.__class__.__name__}")
        
        token.on_consumed()
        self.on_token_removed(token)
        return token
    
    def peek_token(self, token_type: Type[Token] = None) -> Optional[Token]:
        """
        Returns a token without removing it from the place.
        
        Args:
            token_type: Optional specific token type to find
            
        Returns:
            The token if found, None otherwise
        """
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
        """
        Counts tokens of a specific type.
        
        Args:
            token_type: Token type to count, or None for all tokens
            
        Returns:
            Number of matching tokens
        """
        if token_type is None:
            return len(self._tokens)
        else:
            return sum(1 for token in self._tokens if isinstance(token, token_type))
    
    def on_token_added(self, token: Token):
        """Called when a token is added to this place."""
        pass
    
    def on_token_removed(self, token: Token):
        """Called when a token is removed from this place."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(tokens={self.token_count})"


@dataclass
class Arc:
    """Represents a connection between a place and transition with metadata."""
    place: Type[Place]
    weight: int = 1
    token_type: Optional[Type[Token]] = None
    label: Optional[str] = None
    
    def __post_init__(self):
        if self.label is None and self.place is not None:
            self.label = self.place.__name__.lower()
        elif self.label is None:
            self.label = "unnamed_arc"


# Flexible token access system
class ConsumedTokens:
    """Flexible accessor for consumed tokens supporting multiple access patterns."""
    
    def __init__(self, tokens_by_label: Dict[str, List[Token]], 
                 arcs_definition: Union[List[Arc], Dict[str, Arc]]):
        self._tokens_by_label = tokens_by_label
        self._arcs_definition = arcs_definition
        
        # Create lookup mappings
        self._tokens_by_index = list(tokens_by_label.values())
        self._tokens_by_place = {}
        
        if isinstance(arcs_definition, dict):
            for label, arc in arcs_definition.items():
                self._tokens_by_place[arc.place] = tokens_by_label[label]
        else:  # list
            for i, arc in enumerate(arcs_definition):
                label = f"arc_{i}"
                self._tokens_by_place[arc.place] = tokens_by_label[label]
    
    @property
    def key(self):
        """Access by dictionary key (only works if INPUT_ARCS was a dict)."""
        if not isinstance(self._arcs_definition, dict):
            raise ValueError("Cannot use .key accessor when INPUT_ARCS is a list")
        return self._tokens_by_label
    
    @property
    def index(self):
        """Access by list index."""
        return self._tokens_by_index
    
    @property
    def place(self):
        """Access by place type."""
        return self._tokens_by_place


class OutputProducer:
    """Flexible producer for output tokens supporting multiple access patterns."""
    
    def __init__(self, ctx, arcs_definition: Union[List[Arc], Dict[str, Arc]]):
        self._ctx = ctx
        self._arcs_definition = arcs_definition
        
        # Create lookup mappings
        if isinstance(arcs_definition, dict):
            self._arcs_by_key = arcs_definition
            self._arcs_by_index = list(arcs_definition.values())
        else:  # list
            self._arcs_by_key = {f"arc_{i}": arc for i, arc in enumerate(arcs_definition)}
            self._arcs_by_index = arcs_definition
        
        self._arcs_by_place = {}
        for arc in (arcs_definition.values() if isinstance(arcs_definition, dict) else arcs_definition):
            self._arcs_by_place[arc.place] = arc
    
    @property
    def key(self):
        """Access by dictionary key (only works if OUTPUT_ARCS was a dict)."""
        if not isinstance(self._arcs_definition, dict):
            raise ValueError("Cannot use .key accessor when OUTPUT_ARCS is a list")
        return OutputKeyAccessor(self._ctx, self._arcs_by_key)
    
    @property
    def index(self):
        """Access by list index."""
        return OutputIndexAccessor(self._ctx, self._arcs_by_index)
    
    @property
    def place(self):
        """Access by place type."""
        return OutputPlaceAccessor(self._ctx, self._arcs_by_place)
    
    def all(self, tokens: List[Token]):
        """Produce tokens to all output arcs."""
        for arc in (self._arcs_definition.values() if isinstance(self._arcs_definition, dict) else self._arcs_definition):
            for token in tokens:
                self._ctx.produce_token(arc.place, token)


class OutputKeyAccessor:
    def __init__(self, ctx, arcs_by_key):
        self._ctx = ctx
        self._arcs_by_key = arcs_by_key
    
    def __getitem__(self, key):
        if isinstance(key, list):
            return OutputMultiArcProducer(self._ctx, [self._arcs_by_key[k] for k in key])
        return OutputArcProducer(self._ctx, self._arcs_by_key[key])


class OutputIndexAccessor:
    def __init__(self, ctx, arcs_by_index):
        self._ctx = ctx
        self._arcs_by_index = arcs_by_index
    
    def __getitem__(self, index):
        if isinstance(index, list):
            return OutputMultiArcProducer(self._ctx, [self._arcs_by_index[i] for i in index])
        return OutputArcProducer(self._ctx, self._arcs_by_index[index])


class OutputPlaceAccessor:
    def __init__(self, ctx, arcs_by_place):
        self._ctx = ctx
        self._arcs_by_place = arcs_by_place
    
    def __getitem__(self, place_type):
        if isinstance(place_type, list):
            return OutputMultiArcProducer(self._ctx, [self._arcs_by_place[p] for p in place_type])
        return OutputArcProducer(self._ctx, self._arcs_by_place[place_type])


class OutputArcProducer:
    def __init__(self, ctx, arc):
        self._ctx = ctx
        self._arc = arc
    
    def __call__(self, tokens: List[Token]):
        """Produce tokens to this arc."""
        for token in tokens:
            self._ctx.produce_token(self._arc.place, token)


class OutputMultiArcProducer:
    def __init__(self, ctx, arcs):
        self._ctx = ctx
        self._arcs = arcs
    
    def __call__(self, tokens: List[Token]):
        """Produce tokens to multiple arcs."""
        for arc in self._arcs:
            for token in tokens:
                self._ctx.produce_token(arc.place, token)


# Core Transition System
class Transition(ABC):
    """
    Base class for all transitions in the Petri net system.
    Transitions consume tokens from input places and produce tokens to output places
    when their conditions are met. They encapsulate the business logic of the system.
    Now also provides declarative arc definitions and automatic token handling.
    """
    PRIORITY: int = 0  # Higher values fire first

    def __init__(self):
        self._fire_count = 0
        self._last_fired = None
        self._peeked_tokens = {}      # arc_label -> List[Token]
        self._consumed_tokens = {}    # arc_label -> List[Token]

    def input_arcs(self):
        """
        Override this method to define input arcs.
        Can return either:
        - Dict[str, Arc]: {"label": Arc(Place)}
        - List[Arc]: [Arc(Place1), Arc(Place2)]
        """
        return {}

    def output_arcs(self):
        """
        Override this method to define output arcs.
        Can return either:
        - Dict[str, Arc]: {"label": Arc(Place)}
        - List[Arc]: [Arc(Place1), Arc(Place2)]
        """
        return {}

    def _get_input_arcs(self):
        return self.input_arcs()

    def _get_output_arcs(self):
        return self.output_arcs()

    def peek(self, arc_label: str) -> list:
        if arc_label not in self._peeked_tokens:
            try:
                arc = self._get_arc_by_label(arc_label, self._get_input_arcs())
                place = self._get_place_for_arc(arc)
                self._peeked_tokens[arc_label] = place.tokens.copy()
            except Exception:
                self._peeked_tokens[arc_label] = []
        return self._peeked_tokens[arc_label]

    def consume(self, arc_label: str, token):
        if arc_label not in self._consumed_tokens:
            self._consumed_tokens[arc_label] = []
        available_tokens = self.peek(arc_label)
        if token not in available_tokens:
            raise InsufficientTokens(f"Token {token} not available for arc {arc_label}")
        self._consumed_tokens[arc_label].append(token)

    def consume_matching(self, arc_label: str, predicate) -> object:
        available_tokens = self.peek(arc_label)
        for token in available_tokens:
            if predicate(token):
                self.consume(arc_label, token)
                return token
        raise InsufficientTokens(f"No matching token for arc {arc_label}")

    def consume_matching_pair(self, arc1_label: str, arc2_label: str, matcher) -> tuple:
        for token1 in self.peek(arc1_label):
            for token2 in self.peek(arc2_label):
                if matcher(token1, token2):
                    self.consume(arc1_label, token1)
                    self.consume(arc2_label, token2)
                    return (token1, token2)
        raise InsufficientTokens(f"No matching pair found between {arc1_label} and {arc2_label}")

    def _clear_peek_consume_state(self):
        self._peeked_tokens.clear()
        self._consumed_tokens.clear()

    def _get_arc_by_label(self, label: str, arcs):
        if isinstance(arcs, dict):
            return arcs[label]
        else:
            try:
                index = int(label)
                return arcs[index]
            except (ValueError, IndexError):
                raise KeyError(f"Invalid arc label: {label}")

    def _get_place_for_arc(self, arc):
        return self._current_ctx._net._get_place(arc.place)

    def _normalize_arcs_for_processing(self, arcs):
        if isinstance(arcs, list):
            return {str(i): arc for i, arc in enumerate(arcs)}
        return arcs

    @property
    def fire_count(self) -> int:
        return self._fire_count

    @property
    def last_fired(self) -> Optional['datetime']:
        return self._last_fired

    def fire(self, ctx):
        pass

    def can_fire(self, ctx):
        self._current_ctx = ctx
        self._clear_peek_consume_state()
        try:
            return self.guard(ctx)
        finally:
            pass

    def guard(self, ctx):
        input_arcs = self._normalize_arcs_for_processing(self._get_input_arcs())
        for label, arc in input_arcs.items():
            if len(self.peek(label)) < arc.weight:
                return False
        return True

    def _execute_fire(self, ctx):
        self._current_ctx = ctx
        try:
            self._clear_peek_consume_state()
            if not self.can_fire(ctx):
                return False
            consumed_for_fire = {}
            for arc_label, tokens in self._consumed_tokens.items():
                arc = self._get_arc_by_label(arc_label, self._get_input_arcs())
                place = self._get_place_for_arc(arc)
                consumed_for_fire[arc_label] = []
                for token in tokens:
                    if token in place._tokens:
                        place._tokens.remove(token)
                        token.on_consumed()
                        consumed_for_fire[arc_label].append(token)
            if not self._consumed_tokens:
                consumed_for_fire = self._default_consume(ctx)
            output_producer = OutputProducer(ctx, self._get_output_arcs())
            self.on_before_fire(ctx)
            self.fire(ctx, consumed_for_fire, output_producer)
            self._fire_count += 1
            self._last_fired = datetime.now()
            self.on_after_fire(ctx)
            return True
        finally:
            self._clear_peek_consume_state()
            self._current_ctx = None

    def _default_consume(self, ctx):
        input_arcs = self._normalize_arcs_for_processing(self._get_input_arcs())
        consumed_tokens_raw = {}
        for label, arc in input_arcs.items():
            tokens = []
            for _ in range(arc.weight):
                token = ctx.consume_token(arc.place, arc.token_type)
                tokens.append(token)
            consumed_tokens_raw[label] = tokens
        return consumed_tokens_raw

    def fire(self, ctx, consumed_tokens: dict, produce):
        pass

    def on_before_fire(self, ctx):
        pass

    def on_after_fire(self, ctx):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(fired={self.fire_count})"

# Context and Messaging
@dataclass
class PetriNetContext:
    """
    Context object passed to transitions providing access to the Petri net state.
    
    This object provides a safe interface for transitions to interact with
    places and tokens without directly accessing the net's internal state.
    """
    
    _net: 'PetriNet'
    user_context: Any = None
    msg: Optional[Any] = None
    
    def tokens_in_place(self, place_type: Type[Place]) -> List[Token]:
        """Returns a copy of tokens in the specified place."""
        place = self._net._get_place(place_type)
        return place.tokens
    
    def token_count(self, place_type: Type[Place], token_type: Type[Token] = None) -> int:
        """Returns the count of tokens in the specified place."""
        place = self._net._get_place(place_type)
        return place.count_tokens(token_type)
    
    def peek_token(self, place_type: Type[Place], token_type: Type[Token] = None) -> Optional[Token]:
        """Peeks at a token without consuming it."""
        place = self._net._get_place(place_type)
        return place.peek_token(token_type)
    
    def consume_token(self, place_type: Type[Place], token_type: Type[Token] = None) -> Token:
        """Consumes a token from the specified place."""
        place = self._net._get_place(place_type)
        return place.remove_token(token_type)
    
    def produce_token(self, place_type: Type[Place], token: Token):
        """Produces a token to the specified place."""
        place = self._net._get_place(place_type)
        place.add_token(token)
    
    def send_message(self, message: Any):
        """Sends a message to the Petri net."""
        self._net.send_message(message)
    
    def log(self, message: str):
        """Logs a message."""
        self._net.log(message)


# Declarative Composition System
class DeclarativeMeta(type):
    """Metaclass that handles declarative composition for interfaces and nets."""
    
    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class normally first
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Find all nested classes and organize them
        cls._discovered_places = []
        cls._discovered_transitions = []
        cls._discovered_interfaces = []
        
        for attr_name, attr_value in namespace.items():
            if inspect.isclass(attr_value):
                # Set a reference to the outer class on nested classes
                attr_value._outer_class = cls
                
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


# Combined metaclass that inherits from both ABCMeta and DeclarativeMeta
class DeclarativeABCMeta(DeclarativeMeta, ABCMeta):
    """Combined metaclass that supports both ABC and declarative composition."""
    pass


class Interface(ABC, metaclass=DeclarativeABCMeta):
    """Base class for interface nets using declarative composition."""
    
    def __init__(self):
        self._auto_build()
    
    def _auto_build(self):
        """Automatically build the interface."""
        if hasattr(self, 'build'):
            self.build()
    
    def build(self):
        """Optional - only override if you need custom build logic."""
        pass
    
    def get_all_places(self) -> List[Type[Place]]:
        """Returns all place classes in this interface."""
        return self._discovered_places
    
    def get_all_transitions(self) -> List[Type[Transition]]:
        """Returns all transition classes in this interface."""
        return self._discovered_transitions
    
    def get_all_interfaces(self) -> List[Type['Interface']]:
        """Returns all nested interface classes."""
        return self._discovered_interfaces


# Main Petri Net Engine
class PetriNet(metaclass=DeclarativeABCMeta):
    """
    Main Petri net execution engine with declarative composition.
    
    Manages places, transitions, and the overall execution flow of the system.
    Provides message passing, external routing, and visualization capabilities.
    """
    
    MESSAGE_PRIORITY = 2
    INTERNAL_MESSAGE_PRIORITY = 1
    ERROR_PRIORITY = 0
    
    def __init__(self, context: Any = None, log_fn=print, filter_fn=None, trap_fn=None, on_error_fn=None):
        self._places: Dict[Type[Place], Place] = {}
        self._transitions: List[Transition] = []
        self._message_queue = PriorityQueue()
        
        self._log_fn = log_fn or (lambda x: None)
        self._filter_fn = filter_fn or (lambda ctx, msg: False)
        self._trap_fn = trap_fn or (lambda ctx: None)
        self._on_error_fn = on_error_fn or (lambda ctx, exc: None)
        
        self._is_finished = False
        self._tick_count = 0
        self._firing_log = []
        self._transition_log = []
        
        self._ctx = PetriNetContext(self, context)
        self._auto_build()
    
    def _auto_build(self):
        """Automatically build the net from discovered components."""
        # Add all discovered interfaces first
        for interface_class in self._discovered_interfaces:
            self.add_interface(interface_class)
        
        # Add all discovered places
        for place_class in self._discovered_places:
            self.add_place(place_class)
        
        # Add all discovered transitions
        for transition_class in self._discovered_transitions:
            self.add_transition(transition_class())
        
        # Custom build logic if needed
        if hasattr(self, 'build'):
            self.build()
    
    def build(self):
        """Optional - only override if you need custom build logic."""
        pass
    
    @property
    def context(self) -> PetriNetContext:
        """Returns the current context."""
        return self._ctx
    
    @property
    def is_finished(self) -> bool:
        """Returns True if the net has finished execution."""
        return self._is_finished
    
    @property
    def tick_count(self) -> int:
        """Returns the number of ticks processed."""
        return self._tick_count
    
    def log(self, message: str):
        """Logs a message."""
        self._log_fn(f"[Tick {self._tick_count}] {message}")
    
    def add_interface(self, interface_class: Type[Interface]):
        """Add an interface by flattening all its components."""
        interface_instance = interface_class()
        
        # Add all places from the interface
        for place_class in interface_instance.get_all_places():
            if place_class not in self._places:
                self.add_place(place_class)
        
        # Add all transitions from the interface
        for transition_class in interface_instance.get_all_transitions():
            self.add_transition(transition_class())
        
        # Add nested interfaces recursively
        for nested_interface_class in interface_instance.get_all_interfaces():
            self.add_interface(nested_interface_class)
        
        return interface_instance
    
    def add_place(self, place_class: Type[Place]):
        """Add a place to the net."""
        if place_class not in self._places:
            self._places[place_class] = place_class()
            self.log(f"Added place: {place_class.__name__}")
        return self._places[place_class]
    
    def add_transition(self, transition: Union[Transition, Type[Transition]]):
        """Add a transition to the net."""
        if isinstance(transition, type):
            transition = transition()
        
        if not isinstance(transition, Transition):
            raise InvalidTransitionType(f"{transition} is not a valid Transition")
        
        self._transitions.append(transition)
        self.log(f"Added transition: {transition.__class__.__name__}")
        return transition
    
    def attach_output_handler(self, place_type: Type[Place], handler):
        """
        Attach an external handler to a place for IO processing.
        
        The handler will be called with (token, place) arguments whenever
        a token is added to the specified place. This is useful for implementing
        IO boundaries where external systems need to be notified.
        
        Args:
            place_type: The place class to attach the handler to
            handler: Function that takes (token, place) arguments
        """
        place = self._get_place(place_type)
        place.add_token_callback(handler)
        self.log(f"Attached output handler to {place_type.__name__}")
    
    def detach_output_handler(self, place_type: Type[Place], handler):
        """
        Detach an external handler from a place.
        
        Args:
            place_type: The place class to detach the handler from
            handler: The handler function to remove
        """
        place = self._get_place(place_type)
        place.remove_token_callback(handler)
        self.log(f"Detached output handler from {place_type.__name__}")
    
    def _get_place(self, place_type: Type[Place]) -> Place:
        """Gets a place by type, raising an error if not found."""
        if place_type not in self._places:
            raise KeyError(f"Place {place_type.__name__} not found")
        return self._places[place_type]
    
    def produce_token(self, place_type: Type[Place], token: Token):
        """Produces a token to the specified place."""
        place = self._get_place(place_type)
        place.add_token(token)
        self.log(f"Produced {token} to {place_type.__name__}")
    
    def consume_token(self, place_type: Type[Place], token_type: Type[Token] = None) -> Token:
        """Consumes a token from the specified place."""
        place = self._get_place(place_type)
        token = place.remove_token(token_type)
        self.log(f"Consumed {token} from {place_type.__name__}")
        return token
    
    def send_message(self, message: Any):
        """Sends a message to the net."""
        try:
            if message and self._filter_fn(self._ctx, message):
                return
            
            if isinstance(message, Exception):
                priority = self.ERROR_PRIORITY
            else:
                priority = self.MESSAGE_PRIORITY
            
            self._message_queue.put((priority, message))
        except Exception as e:
            self._message_queue.put((self.ERROR_PRIORITY, e))
    
    def send_message_to_place(self, place_type: Type[Place], token: Token):
        """Sends a token directly to a specific place."""
        self.produce_token(place_type, token)
        self.log(f"Routed message to {place_type.__name__}: {token}")
    
    def tick(self, timeout: Optional[float] = 0):
        """
        Executes one tick of the Petri net.
        
        Args:
            timeout: Message queue timeout (0=non-blocking, None=blocking forever)
        """
        self._tick_count += 1
        
        # Process messages
        try:
            if timeout == 0:
                try:
                    priority, message = self._message_queue.get_nowait()
                    self._ctx.msg = message
                except Empty:
                    self._ctx.msg = None
            else:
                try:
                    priority, message = self._message_queue.get(timeout=timeout)
                    self._ctx.msg = message
                except Empty:
                    self._ctx.msg = None
        except Exception as e:
            self._ctx.msg = e
        
        # Handle exceptions
        if isinstance(self._ctx.msg, Exception):
            next_action = self._on_error_fn(self._ctx, self._ctx.msg)
            if next_action is None:
                raise self._ctx.msg
        
        # Find fireable transitions
        fireable_transitions = []
        for transition in self._transitions:
            try:               
                if transition.can_fire(self._ctx):
                    fireable_transitions.append(transition)
            except Exception as e:
                self.log(f"Error checking if {transition} can fire: {e}")
                # Clear any partial context that might have been set
                if hasattr(transition, '_current_ctx'):
                    transition._current_ctx = None
                continue
        
        # Sort by priority
        fireable_transitions.sort(key=lambda t: t.PRIORITY, reverse=True)
        
        # Fire transitions
        fired_any = False
        for transition in fireable_transitions:
            try:
                result = transition._execute_fire(self._ctx)
                if result is not False:  # ArcBasedTransitions return False if they can't fire
                    self._firing_log.append((self._tick_count, transition.__class__.__name__))
                    self.log(f"Fired transition: {transition.__class__.__name__}")
                    fired_any = True
                
            except Exception as e:
                self.log(f"Error firing {transition}: {e}")
                next_action = self._on_error_fn(self._ctx, e)
                if next_action is None:
                    raise e
        
        # Handle unhandled messages
        if self._ctx.msg and not fired_any:
            self._trap_fn(self._ctx)
        
        # Clear message
        self._ctx.msg = None
        
        # Check termination conditions
        if self._should_terminate():
            self._is_finished = True
            raise PetriNetComplete("Petri net execution completed")
    
    def _should_terminate(self) -> bool:
        """
        Determines if the net should terminate.
        
        Override this method to implement custom termination logic.
        Default implementation never terminates.
        """
        return False
    
    def run_until_complete(self, max_ticks: int = 1000):
        """
        Runs the net until completion or max ticks reached.
        
        Args:
            max_ticks: Maximum number of ticks to process
        """
        for _ in range(max_ticks):
            try:
                self.tick()
            except PetriNetComplete:
                break
            except Exception as e:
                self.log(f"Execution stopped due to error: {e}")
                raise
        else:
            self.log(f"Execution stopped after {max_ticks} ticks")
    
    def start_non_blocking(self, max_ticks: int = 1000):
        """
        Starts non-blocking execution.
        
        Args:
            max_ticks: Maximum number of ticks to process
        """
        for _ in range(max_ticks):
            try:
                self.tick(timeout=0)
            except PetriNetComplete:
                break
            except Empty:
                raise NonBlockingStalled("Non-blocking execution stalled")
            except Exception as e:
                self.log(f"Non-blocking execution stopped due to error: {e}")
                raise
    
    def reset(self):
        """Resets the net to initial state."""
        # Clear all places
        for place in self._places.values():
            place._tokens.clear()
        
        # Reset transitions
        for transition in self._transitions:
            transition._fire_count = 0
            transition._last_fired = None
        
        # Clear message queue
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except Empty:
                break
        
        self._is_finished = False
        self._tick_count = 0
        self._firing_log.clear()
        self._transition_log.clear()
        self.log("Net reset")
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current net state."""
        return {
            'tick_count': self._tick_count,
            'is_finished': self._is_finished,
            'places': {
                place_type.__name__: {
                    'token_count': place.token_count,
                    'tokens': [repr(token) for token in place.tokens]
                }
                for place_type, place in self._places.items()
            },
            'transitions': {
                transition.__class__.__name__: {
                    'fire_count': transition.fire_count,
                    'last_fired': transition.last_fired
                }
                for transition in self._transitions
            },
            'firing_log': self._firing_log
        }
    
    def generate_mermaid_diagram(self) -> str:
        """
        Generates a Mermaid diagram representing the Petri net structure.
        
        Returns:
            A string containing the Mermaid diagram
        """
        lines = ["graph TD"]
        
        # Add places as circles
        for place_type in self._places.keys():
            place_name = place_type.__name__
            token_count = self._places[place_type].token_count
            # Use circle syntax for places
            lines.append(f'    {place_name}(("{place_name}<br/>({token_count} tokens)"))')
        
        # Add transitions as rectangles and their connections
        for transition in self._transitions:
            transition_name = transition.__class__.__name__
            fire_count = transition.fire_count
            # Use rectangle syntax for transitions
            lines.append(f'    {transition_name}["{transition_name}<br/>(fired {fire_count}x)"]')
            
            # Add connections if this is a Transition
            if isinstance(transition, Transition):
                # Input arcs
                input_arcs = transition._normalize_arcs_for_processing(transition.input_arcs())
                for label, arc in input_arcs.items():
                    place_name = arc.place.__name__
                    if arc.weight > 1:
                        lines.append(f'    {place_name} -->|"{arc.weight}"| {transition_name}')
                    else:
                        lines.append(f'    {place_name} --> {transition_name}')
                
                # Output arcs
                output_arcs = transition._normalize_arcs_for_processing(transition.output_arcs())
                for label, arc in output_arcs.items():
                    place_name = arc.place.__name__
                    if arc.weight > 1:
                        lines.append(f'    {transition_name} -->|"{arc.weight}"| {place_name}')
                    else:
                        lines.append(f'    {transition_name} --> {place_name}')
        
        return "\n".join(lines)
    
    def save_mermaid_diagram(self, filename: str):
        """
        Saves the Mermaid diagram to a file.
        
        Args:
            filename: Path to save the diagram file
        """
        with open(filename, 'w') as f:
            f.write(self.generate_mermaid_diagram())
    
    def __repr__(self):
        return f"PetriNet(places={len(self._places)}, transitions={len(self._transitions)}, tick={self._tick_count})"


# Utility Functions
def create_simple_token(data: Any = None) -> Token:
    """Creates a simple token with the given data."""
    class SimpleToken(Token):
        pass
    return SimpleToken(data)


def create_resource_pool(place_type: Type[Place], token_type: Type[Token], count: int) -> List[Token]:
    """Creates a list of tokens for initializing a resource pool."""
    return [token_type() for _ in range(count)]