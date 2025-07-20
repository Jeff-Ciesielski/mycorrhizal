#!/usr/bin/env python3
"""
Cordyceps Gevent Core - Autonomous Queue-Based Petri Net Library

This is a complete rewrite using gevent for autonomous execution without colored functions.
Each transition runs as an independent greenlet waiting on input queues.
"""

import gevent
from gevent import Greenlet, spawn, sleep
from gevent.queue import Queue, Empty, Full
from gevent.event import Event
import inspect
import traceback
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from collections import defaultdict
from enum import Enum
import weakref


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
    
    Places are now backed by gevent queues and operate autonomously.
    Each place has exactly one associated transition (one-to-one mapping).
    """
    
    MAX_CAPACITY: Optional[int] = None
    ACCEPTED_TOKEN_TYPES: Optional[Set[Type[Token]]] = None
    
    def __init__(self):
        self._queue: Queue = None  # Will be set during net construction
        self._total_tokens_processed = 0
        self._net: Optional['PetriNet'] = None
        self._qualified_name: Optional[str] = None
        self._greenlet: Optional[Greenlet] = None
        
    def _attach_to_net(self, net: 'PetriNet'):
        """Called when place is attached to a net."""
        self._net = net
        # Create queue with capacity if specified
        maxsize = self.MAX_CAPACITY if self.MAX_CAPACITY is not None else None
        self._queue = Queue(maxsize=maxsize)
        
    def _set_qualified_name(self, qualified_name: str):
        """Set the fully qualified name for this place."""
        self._qualified_name = qualified_name
        
    @property
    def net(self) -> "PetriNet":
        return self._net
        
    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name for this place."""
        return self._qualified_name or self.__class__.__name__
        
    @property
    def token_count(self) -> int:
        """Returns the current number of tokens in this place."""
        return self._queue.qsize() if self._queue else 0
    
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
    
    def can_accept_token(self, token: Token) -> bool:
        """Check if this place can accept the given token."""
        if self.is_full:
            return False
            
        if self.ACCEPTED_TOKEN_TYPES is not None:
            return type(token) in self.ACCEPTED_TOKEN_TYPES
            
        return True
    
    def add_token(self, token: Token, block: bool = True, timeout: Optional[float] = None):
        """
        Add a token to this place's queue.
        
        Args:
            token: Token to add
            block: Whether to block if queue is full
            timeout: Timeout for blocking operation
        
        Raises:
            InvalidTokenType: If token type is not accepted
            Full: If queue is full and block=False
        """
        if not self.can_accept_token(token):
            if self.is_full:
                raise InsufficientTokens(f"Place {self.qualified_name} is at capacity")
            else:
                raise InvalidTokenType(f"Place {self.qualified_name} cannot accept token type {type(token)}")
        
        self._queue.put(token, block=block, timeout=timeout)
        self._total_tokens_processed += 1
        token.on_produced()
        self.on_token_added(token)
    
    def get_token(self, block: bool = True, timeout: Optional[float] = None) -> Token:
        """
        Get a token from this place's queue.
        
        Args:
            block: Whether to block if queue is empty
            timeout: Timeout for blocking operation
            
        Returns:
            Token from the queue
            
        Raises:
            Empty: If queue is empty and block=False
        """
        token = self._queue.get(block=block, timeout=timeout)
        token.on_consumed()
        self.on_token_removed(token)
        return token
    
    def peek_token(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Token]:
        """Peek at the next token without removing it."""
        # gevent.queue doesn't have peek, so we get and put back
        try:
            token = self._queue.get(block=block, timeout=timeout)
            # Put it back immediately
            self._queue.put(token, block=False)
            return token
        except (Empty, Full):
            return None
    
    def on_token_added(self, token: Token):
        """Called when a token is added to this place."""
        pass
    
    def on_token_removed(self, token: Token):
        """Called when a token is removed from this place."""
        pass
    
    def __repr__(self):
        return f"{self.qualified_name}(tokens={self.token_count})"


class IOOutputPlace(Place):
    """
    Output IO places that send tokens to external systems.
    
    These places run autonomous greenlets that continuously drain their queues
    and send tokens to external systems.
    """
    
    def _start_output_task(self):
        """Start the output processing greenlet."""
        if self._greenlet is None:
            self._greenlet = spawn(self._output_loop)
            if self._net:
                self._net._running_tasks.add(self._greenlet)
    
    def _output_loop(self):
        """Main output loop - continuously processes tokens from queue."""
        try:
            self.on_output_start()
            
            while self._net and self._net._is_running.is_set():
                try:
                    # Get token with timeout to check for shutdown
                    token = self._queue.get(timeout=0.1)
                    
                    # Process the token
                    error_token = self.on_token_added(token)
                    
                    if error_token is None:
                        # Success
                        self.on_io_success(token)
                    else:
                        # Failed - put error token back in queue
                        self._queue.put(error_token, block=False)
                        self.on_io_error(token, error_token)
                        
                except Empty:
                    continue  # Check for shutdown
                except Exception as e:
                    self.log(f"Unexpected error in IO output place {self.qualified_name}: {e}")
                    self.on_io_error(token if 'token' in locals() else None, None)
            
            self.on_output_stop()
            
        except Exception as e:
            self.log(f"Fatal error in output loop for {self.qualified_name}: {e}")
        finally:
            if self._net and self._greenlet in self._net._running_tasks:
                self._net._running_tasks.discard(self._greenlet)
    
    def on_token_added(self, token: Token) -> Optional[Token]:
        """
        Handle outgoing token. Override this for output places.
        
        Returns:
            None if successful (token is consumed)
            Error token if failed (replaces original token)
        """
        # Default implementation just logs and succeeds
        self.log(f"IO Output: {token}")
        return None
    
    def on_output_start(self):
        """Called when output processing starts."""
        pass
    
    def on_output_stop(self):
        """Called when output processing stops."""
        pass
    
    def on_io_success(self, token: Token):
        """Called when IO operation succeeds."""
        pass
    
    def on_io_error(self, token: Token, error_token: Optional[Token]):
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
    Input IO places that receive tokens from external systems.
    
    These places run autonomous greenlets that produce tokens into their queues
    from external events/sources.
    """
    
    def _start_input_task(self):
        """Start the input processing greenlet."""
        if self._greenlet is None:
            self._greenlet = spawn(self._input_loop)
            if self._net:
                self._net._running_tasks.add(self._greenlet)
    
    def _input_loop(self):
        """Main input loop - calls on_input() and puts tokens in queue."""
        try:
            self.on_input_start()
            
            while self._net and self._net._is_running.is_set():
                try:
                    # Call user-defined input handler
                    result = self.on_input()
                    
                    if isinstance(result, Token):
                        self._queue.put(result)
                        result.on_produced()
                        self.on_token_added(result)
                    elif isinstance(result, list) and all(isinstance(x, Token) for x in result):
                        for token in result:
                            self._queue.put(token)
                            token.on_produced()
                            self.on_token_added(token)
                    elif result is None:
                        pass
                    else:
                        raise TypeError(f"IOInput Places must produce Tokens or Lists of Tokens, got: {result}")
                    
                    # Small delay to prevent busy waiting
                    sleep(0.01)
                    
                except Exception as e:
                    self.log(f"Error in input loop for {self.qualified_name}: {e}")
                    sleep(0.1)  # Back off on error
            
            self.on_input_stop()
            
        except Exception as e:
            self.log(f"Fatal error in input loop for {self.qualified_name}: {e}")
        finally:
            if self._net and self._greenlet in self._net._running_tasks:
                self._net._running_tasks.discard(self._greenlet)
    
    def on_input_start(self):
        """Called when input starts."""
        pass
    
    def on_input(self) -> Union[Token, List[Token], None]:
        """
        Main input handler - override this to generate tokens.
        
        This method is called repeatedly until shutdown.
        Return tokens to add to this place's queue.
        """
        # Default implementation waits for shutdown
        sleep(1.0)
        return None
    
    def on_input_stop(self):
        """Called when input stops."""
        pass
    
    def log(self, message: str):
        """Log a message."""
        if self._net:
            self._net.log(message)
        else:
            print(message)


@dataclass
class Arc:
    """Represents a connection between a place and transition with metadata."""
    place: Union[Type[Place], str]  # Place reference - class or qualified name string
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


class DispatchPolicy(Enum):
    """Policies for when transitions should fire."""
    ANY = "any"    # Fire when any input place has required tokens
    ALL = "all"    # Fire when all input places have required tokens


class PolicyDispatcher:
    """
    Manages token collection and transition firing based on policy.
    
    Replaces the guard system with policy-based dispatching.
    """
    
    def __init__(self, transition: 'Transition', policy: DispatchPolicy):
        self.transition = transition
        self.policy = policy
        self.input_places: Dict[PlaceName, Place] = {}
        self.input_weights: Dict[PlaceName, int] = {}
        self.collector_queues: Dict[PlaceName, Queue] = {}
        self._greenlet: Optional[Greenlet] = None
    
    def add_input(self, label: PlaceName, place: Place, weight: int):
        """Add an input place with its weight."""
        self.input_places[label] = place
        self.input_weights[label] = weight
        
        if weight > 1:
            # Create collector queue for batching
            self.collector_queues[label] = Queue()
    
    def start(self):
        """Start the dispatcher greenlet."""
        if self._greenlet is None:
            self._greenlet = spawn(self._dispatch_loop)
            if self.transition._net:
                self.transition._net._running_tasks.add(self._greenlet)
    
    def _dispatch_loop(self):
        """Main dispatch loop."""
        try:
            # Start collector tasks for weighted inputs
            collector_greenlets = []
            for label, weight in self.input_weights.items():
                if weight > 1:
                    collector = spawn(self._collector_loop, label, weight)
                    collector_greenlets.append(collector)
                    if self.transition._net:
                        self.transition._net._running_tasks.add(collector)
            
            while self.transition._net and self.transition._net._is_running.is_set():
                try:
                    # Collect tokens based on policy
                    token_batch = self._collect_tokens()
                    
                    if token_batch:
                        # Forward to transition
                        self.transition._input_queue.put(token_batch)
                    
                    sleep(0.01)  # Small delay
                    
                except Exception as e:
                    if self.transition._net:
                        self.transition._net.log(f"Error in dispatcher for {self.transition.qualified_name}: {e}")
            
            # Clean up collector tasks
            for collector in collector_greenlets:
                collector.kill()
                
        except Exception as e:
            if self.transition._net:
                self.transition._net.log(f"Fatal error in dispatcher for {self.transition.qualified_name}: {e}")
        finally:
            if self.transition._net and self._greenlet in self.transition._net._running_tasks:
                self.transition._net._running_tasks.discard(self._greenlet)
    
    def _collector_loop(self, label: PlaceName, weight: int):
        """Collector task for batching tokens from weighted arcs."""
        place = self.input_places[label]
        collector_queue = self.collector_queues[label]
        
        try:
            while self.transition._net and self.transition._net._is_running.is_set():
                batch = []
                
                # Collect required number of tokens
                for _ in range(weight):
                    try:
                        token = place.get_token(timeout=0.1)
                        batch.append(token)
                    except Empty:
                        # Put back any partial batch
                        for token in reversed(batch):
                            place._queue.put(token, block=False)
                        batch = []
                        break
                
                if batch:
                    # Forward complete batch to collector queue
                    collector_queue.put(batch)
                    
        except Exception as e:
            if self.transition._net:
                self.transition._net.log(f"Error in collector for {self.transition.qualified_name}.{label}: {e}")
    
    def _collect_tokens(self) -> Optional[Dict[PlaceName, List[Token]]]:
        """Collect tokens based on dispatch policy."""
        available = {}
        
        # Check what's available
        for label in self.input_places:
            if label in self.collector_queues:
                # Weighted input - check collector queue
                try:
                    batch = self.collector_queues[label].get(block=False)
                    available[label] = batch
                except Empty:
                    available[label] = []
            else:
                # Regular input - check place queue
                place = self.input_places[label]
                try:
                    token = place.get_token(block=False)
                    available[label] = [token]
                except Empty:
                    available[label] = []
        
        # Apply policy
        if self.policy == DispatchPolicy.ANY:
            # Fire if any input has tokens
            ready = {k: v for k, v in available.items() if v}
            return ready if ready else None
        
        elif self.policy == DispatchPolicy.ALL:
            # Fire only if all inputs have tokens
            if all(available.values()):
                return available
            else:
                # Put back all tokens
                self._put_back_tokens(available)
                return None
        
        return None
    
    def _put_back_tokens(self, token_dict: Dict[PlaceName, List[Token]]):
        """Put tokens back into their places."""
        for label, tokens in token_dict.items():
            if label in self.collector_queues:
                # Put back into collector queue
                if tokens:
                    self.collector_queues[label].put(tokens, block=False)
            else:
                # Put back into place queue
                place = self.input_places[label]
                for token in reversed(tokens):
                    place._queue.put(token, block=False)


# Core Transition System
class Transition(ABC):
    """
    Base class for all transitions in the Petri net system.
    
    In the autonomous model, transitions run as independent greenlets
    waiting on input queues managed by policy dispatchers.
    """
    DISPATCH_POLICY: DispatchPolicy = DispatchPolicy.ALL  # Default policy
    
    def __init__(self):
        self._fire_count = 0
        self._last_fired = None
        self._net: Optional['PetriNet'] = None
        self._qualified_name: Optional[str] = None
        self._interface_path: Optional[str] = None
        self._input_queue: Queue = Queue()
        self._output_places: Dict[PlaceName, Place] = {}
        self._dispatcher: Optional[PolicyDispatcher] = None
        self._greenlet: Optional[Greenlet] = None

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
    def net(self):
        return self._net
        
    @property
    def qualified_name(self) -> str:
        """Get the fully qualified name for this transition."""
        return self._qualified_name or self.__class__.__name__

    def input_arcs(self) -> Dict[str, Arc]:
        """Override this method to define input arcs using PlaceName enums."""
        return {}

    def output_arcs(self) -> Dict[str, Arc]:
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

    def _setup_dispatcher(self):
        """Set up the policy dispatcher for this transition."""
        self._dispatcher = PolicyDispatcher(self, self.DISPATCH_POLICY)
        
        # Add input places to dispatcher
        input_arcs = self.input_arcs()
        for label, arc in input_arcs.items():
            place = self._net._resolve_place_from_arc(arc)
            self._dispatcher.add_input(label, place, arc.weight)
    
    def _start_transition_task(self):
        """Start the transition processing greenlet."""
        if self._greenlet is None:
            self._greenlet = spawn(self._transition_loop)
            if self._net:
                self._net._running_tasks.add(self._greenlet)
        
        if self._dispatcher:
            self._dispatcher.start()
    
    def _transition_loop(self):
        """Main transition loop - processes token batches from dispatcher."""
        try:
            while self._net and self._net._is_running.is_set():
                try:
                    # Wait for token batch from dispatcher
                    token_batch = self._input_queue.get(timeout=0.1)
                    
                    # Fire the transition
                    self._fire_with_tokens(token_batch)
                    
                except Empty:
                    continue  # Check for shutdown
                except Exception as e:
                    self._handle_error(e, token_batch if 'token_batch' in locals() else {})
            
        except Exception as e:
            if self._net:
                self._net.log(f"Fatal error in transition loop for {self.qualified_name}: {e}")
        finally:
            if self._net and self._greenlet in self._net._running_tasks:
                self._net._running_tasks.discard(self._greenlet)
    
    def _fire_with_tokens(self, consumed_tokens: Dict[PlaceName, List[Token]]):
        """Fire the transition with the given consumed tokens."""
        try:
            self.on_before_fire(consumed_tokens)
            
            # Call user-defined firing logic
            output_tokens = self.on_fire(consumed_tokens)
            
            # Produce output tokens
            if output_tokens:
                output_arcs = self.output_arcs()
                for label, tokens in output_tokens.items():
                    if label in output_arcs:
                        place = self._output_places[label]
                        if isinstance(tokens, Token):
                            tokens = [tokens]
                        for token in tokens:
                            place.add_token(token, block=False)
            
            self._fire_count += 1
            self._last_fired = datetime.now()
            
            self.on_after_fire(consumed_tokens, output_tokens)
            
            if self._net:
                self._net.log(f"Fired transition: {self.qualified_name}")
                
        except Exception as e:
            # Handle error and potentially rollback
            self._handle_error(e, consumed_tokens)

    def on_fire(self, consumed: Dict[PlaceName, List[Token]]) -> Optional[Dict[PlaceName, List[Token]]]:
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
    
    def on_before_fire(self, consumed: Dict[PlaceName, List[Token]]):
        """Called before transition fires."""
        pass
    
    def on_after_fire(self, consumed: Dict[PlaceName, List[Token]], produced: Optional[Dict[PlaceName, List[Token]]]):
        """Called after transition fires successfully."""
        pass
    
    def _handle_error(self, error: Exception, consumed_tokens: Dict[PlaceName, List[Token]]):
        """Handle transition errors."""
        if self._net and hasattr(self._net, 'on_transition_error'):
            action = self._net.on_transition_error(self, error, consumed_tokens)
            
            if action == "restart":
                # Put tokens back and continue
                self._rollback_tokens(consumed_tokens)
                return
            elif action == "kill":
                # Shut down the net
                if self._net:
                    self._net.stop()
                return
        
        # Default: re-raise the error
        raise error
    
    def _rollback_tokens(self, consumed_tokens: Dict[PlaceName, List[Token]]):
        """Put consumed tokens back into their places for error recovery."""
        input_arcs = self.input_arcs()
        
        for label, tokens in consumed_tokens.items():
            if label in input_arcs:
                arc = input_arcs[label]
                place = self._net._resolve_place_from_arc(arc)
                
                # Put tokens back in reverse order to maintain FIFO
                for token in reversed(tokens):
                    try:
                        place._queue.put(token, block=False)
                    except Full:
                        # If we can't put it back, we have a problem
                        if self._net:
                            self._net.log(f"WARNING: Could not rollback token to {place.qualified_name}")

    def __repr__(self):
        return f"{self.qualified_name}(fired={self.fire_count})"


# Declarative Composition System (unchanged from original)
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
    """
    Main autonomous Petri net execution engine using gevent.
    
    This creates an autonomous system where transitions run as independent
    greenlets waiting on queues, eliminating the need for central coordination.
    """
    
    def __init__(self, log_fn=print):
        self._places_by_type: Dict[Type[Place], Place] = {}
        self._places_by_qualified_name: Dict[str, Place] = {}
        self._transitions: List[Transition] = []
        self._io_input_places: List[IOInputPlace] = []
        self._io_output_places: List[IOOutputPlace] = []
        
        self._log_fn = log_fn or (lambda x: None)
        
        # Event handling for coordinated shutdown
        self._is_running = Event()
        self._running_tasks: Set[Greenlet] = set()
        
        # Tracking for qualified names and interface boundaries
        self._qualified_name_registry: Dict[str, Any] = {}
        self._interface_hierarchy: Dict[str, List[str]] = {}
        
        # Error handling callback
        self._error_handler: Optional[Callable] = None
        
        self._auto_build()
    
    def set_error_handler(self, handler: Callable[[Transition, Exception, Dict[PlaceName, List[Token]]], str]):
        """
        Set error handler for transition failures.
        
        Handler should return one of: "restart", "kill", or re-raise the exception.
        """
        self._error_handler = handler
    
    def on_transition_error(self, transition: Transition, error: Exception, consumed_tokens: Dict[PlaceName, List[Token]]) -> str:
        """Handle transition errors."""
        if self._error_handler:
            return self._error_handler(transition, error, consumed_tokens)
        
        # Default: log and restart
        self.log(f"Error in transition {transition.qualified_name}: {error}")
        return "restart"
    
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
        
        # Fourth pass: set up autonomous execution
        self._setup_autonomous_execution()
    
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
        
        # Validate input arcs
        try:
            input_arcs = temp_transition.input_arcs()
        except Exception as e:
            raise InterfaceBoundaryViolation(
                f"Transition {transition_qualified_name} has invalid input arcs: {e}"
            )
        
        for label, arc in input_arcs.items():
            self.log(f"Checking input arc {label}: {arc.place}")
            self._validate_arc_boundary(arc.place, transition_interface, transition_qualified_name, "input", 
                                      skip_boundary_check=(not transition_interface))
        
        # Validate output arcs
        try:
            output_arcs = temp_transition.output_arcs()
        except Exception as e:
            raise InterfaceBoundaryViolation(
                f"Transition {transition_qualified_name} has invalid output arcs: {e}"
            )
        
        for label, arc in output_arcs.items():
            self.log(f"Checking output arc {label}: {arc.place}")
            self._validate_arc_boundary(arc.place, transition_interface, transition_qualified_name, "output",
                                      skip_boundary_check=(not transition_interface))
    
    def _validate_arc_boundary(self, place_ref: Union[Type[Place], str], transition_interface: str, 
                             transition_name: str, arc_type: str, skip_boundary_check: bool = False):
        """Validate that an arc doesn't cross interface boundaries or reference missing places."""
        # Resolve the qualified name and ensure the place exists
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
            return
        
        self.log(f"Checking boundary: place {place_qualified_name} vs transition interface {transition_interface}")
        
        # Check if the place is within the same interface or a child interface
        if not self._is_within_interface_boundary(place_qualified_name, transition_interface):
            raise InterfaceBoundaryViolation(
                f"Transition {transition_name} violates interface boundary: "
                f"cannot access {place_qualified_name} from interface {transition_interface}"
            )
    
    def _is_within_interface_boundary(self, place_qualified_name: str, transition_interface: str) -> bool:
        """Allow access to any place that is a direct child or descendant of the transition's interface."""
        self.log(f"Boundary check: {place_qualified_name} within {transition_interface}?")
        
        if place_qualified_name == transition_interface:
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
    
    def _setup_autonomous_execution(self):
        """Set up autonomous execution for all transitions and places."""
        # Set up transitions with their dispatchers and output places
        for transition in self._transitions:
            # Map output places
            output_arcs = transition.output_arcs()
            for label, arc in output_arcs.items():
                place = self._resolve_place_from_arc(arc)
                transition._output_places[label] = place
            
            # Set up dispatcher
            transition._setup_dispatcher()
    
    def log(self, message: str):
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
            raise KeyError(f"Place type {place_type.__name__} not found")
        return self._places_by_type[place_type]
    
    def _get_place_by_qualified_name(self, qualified_name: str) -> Place:
        """Get a place by qualified name."""
        if qualified_name not in self._places_by_qualified_name:
            raise KeyError(f"Place {qualified_name} not found")
        return self._places_by_qualified_name[qualified_name]
    
    def produce_token(self, place_type: Type[Place], token: Token):
        """Produce a token to the specified place."""
        place = self._get_place_by_type(place_type)
        place.add_token(token, block=False)
    
    def start(self):
        """Start the autonomous Petri net execution."""
        if self._is_running.is_set():
            return
        
        self.log(f"Starting Net: {self.__class__.__name__}")
        self._is_running.set()
        
        # Start all IO input places
        for io_input_place in self._io_input_places:
            io_input_place._start_input_task()
        
        # Start all IO output places
        for io_output_place in self._io_output_places:
            io_output_place._start_output_task()
        
        # Start all transitions
        for transition in self._transitions:
            transition._start_transition_task()
        
        self.log(f"Net: {self.__class__.__name__} tasks started")
    
    def stop(self):
        """Stop the autonomous Petri net execution."""
        if not self._is_running.is_set():
            return
        
        self.log(f"Stopping Net: {self.__class__.__name__}")
        self._is_running.clear()
        
        # Wait for all tasks to finish
        gevent.joinall(list(self._running_tasks), timeout=5.0)
        
        # Kill any remaining tasks
        for task in self._running_tasks:
            if not task.dead:
                task.kill()
        
        self._running_tasks.clear()
        self.log(f"Net: {self.__class__.__name__} stopped")
    
    def run_for(self, duration: float):
        """Run the net for a specified duration."""
        self.start()
        try:
            sleep(duration)
        finally:
            self.stop()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current net state."""
        return {
            'is_running': self._is_running.is_set(),
            'places': {
                place.qualified_name: {
                    'token_count': place.token_count,
                    'is_full': place.is_full,
                    'is_empty': place.is_empty
                }
                for place in self._places_by_type.values()
            },
            'transitions': {
                transition.qualified_name: {
                    'fire_count': transition.fire_count,
                    'last_fired': transition.last_fired,
                    'dispatch_policy': transition.DISPATCH_POLICY.value
                }
                for transition in self._transitions
            },
            'running_tasks': len(self._running_tasks)
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
            policy = transition.DISPATCH_POLICY.value
            lines.append(f'    {safe_name}["{qualified_name}<br/>(fired {fire_count}x)<br/>Policy: {policy}"]')

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

            # Input arcs - handle PlaceName enum keys
            try:
                input_arcs = transition.input_arcs()
                for enum_key, arc in input_arcs.items():
                    place = self._resolve_place_from_arc(arc)
                    place_safe_name = place.qualified_name.replace(".", "_")
                    
                    # Create edge label with enum name and weight
                    edge_label = enum_key.name
                    if arc.weight > 1:
                        edge_label += f" ({arc.weight})"
                    
                    lines.append(f'    {place_safe_name} -->|"{edge_label}"| {transition_safe_name}')
            except Exception as e:
                self.log(f"Warning: Could not generate input arcs for {transition.qualified_name}: {e}")

            # Output arcs - handle PlaceName enum keys
            try:
                output_arcs = transition.output_arcs()
                for enum_key, arc in output_arcs.items():
                    place = self._resolve_place_from_arc(arc)
                    place_safe_name = place.qualified_name.replace(".", "_")
                    
                    # Create edge label with enum name
                    edge_label = enum_key.name
                    if arc.weight > 1:
                        edge_label += f" ({arc.weight})"
                    
                    lines.append(f'    {transition_safe_name} -->|"{edge_label}"| {place_safe_name}')
            except Exception as e:
                self.log(f"Warning: Could not generate output arcs for {transition.qualified_name}: {e}")

        return "\n".join(lines)
    
    def save_mermaid_diagram(self, filename: str):
        """Save the Mermaid diagram to a file."""
        with open(filename, 'w') as f:
            f.write(self.generate_mermaid_diagram())
    
    def print_mermaid_diagram(self):
        """Print the Mermaid diagram to console."""
        print(self.generate_mermaid_diagram())
    
    def __repr__(self):
        return f"PetriNet(places={len(self._places_by_type)}, transitions={len(self._transitions)}, running={self._is_running.is_set()})"


# Utility Functions
def create_simple_token(data: Any = None) -> Token:
    """Create a simple token with the given data."""
    class SimpleToken(Token):
        pass
    return SimpleToken(data)