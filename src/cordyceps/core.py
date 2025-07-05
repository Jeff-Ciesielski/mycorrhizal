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
        
    def _attach_to_net(self, net: 'PetriNet'):
        """Called when place is attached to a net."""
        self._net = net
        
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
        """
        Add a token to this place and notify the net.
        
        This is the core event that drives the Petri net execution.
        """
        if not self.can_accept_token(token):
            if self.is_full:
                raise Exception(f"Place {self.__class__.__name__} is at capacity")
            else:
                raise InvalidTokenType(f"Place {self.__class__.__name__} cannot accept token type {type(token)}")
        
        self._tokens.append(token)
        self._total_tokens_processed += 1
        
        await token.on_produced()
        await self.on_token_added(token)
        
        # Notify the net that a token was added - this triggers transition checking
        if self._net:
            await self._net._on_token_added(self, token)
    
    async def remove_token(self, token_type: Type[Token] = None) -> Token:
        """Remove and return a token from this place."""
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
        return f"{self.__class__.__name__}(tokens={self.token_count})"


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
            self.log(f"Unexpected error in IO output place {self.__class__.__name__}: {e}")
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
                    self.log(f"Error in input loop for {self.__class__.__name__}: {e}")
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
    place: Type[Place]
    weight: int = 1
    token_type: Optional[Type[Token]] = None
    label: Optional[str] = None
    
    def __post_init__(self):
        if self.label is None and self.place is not None:
            self.label = self.place.__name__.lower()
        elif self.label is None:
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

    def _attach_to_net(self, net: 'PetriNet'):
        """Called when transition is attached to a net."""
        self._net = net

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

    async def can_fire(self) -> bool:
        """Check if this transition can fire."""
        return await self.guard()

    async def guard(self) -> bool:
        """Default guard: check if all input arcs have sufficient tokens."""
        input_arcs = self._normalize_arcs(self.input_arcs())
        
        for label, arc in input_arcs.items():
            place = self._net._get_place(arc.place)
            if place.count_tokens(arc.token_type) < arc.weight:
                return False
        
        return True

    async def fire(self):
        """
        Fire this transition.
        
        This consumes tokens from input places and produces tokens to output places.
        """
        if not await self.can_fire():
            return False
        
        # Consume input tokens
        input_arcs = self._normalize_arcs(self.input_arcs())
        consumed_tokens = {}
        
        for label, arc in input_arcs.items():
            place = self._net._get_place(arc.place)
            tokens = []
            for _ in range(arc.weight):
                token = await place.remove_token(arc.token_type)
                tokens.append(token)
            consumed_tokens[label] = tokens
        
        # Execute transition logic
        await self.on_before_fire()
        produced_tokens = await self.on_fire(consumed_tokens)
        
        # Produce output tokens
        if produced_tokens:
            output_arcs = self._normalize_arcs(self.output_arcs())
            for label, tokens in produced_tokens.items():
                if label in output_arcs:
                    arc = output_arcs[label]
                    place = self._net._get_place(arc.place)
                    for token in tokens:
                        await place.add_token(token)
        
        self._fire_count += 1
        self._last_fired = datetime.now()
        await self.on_after_fire()
        
        return True

    async def on_fire(self, consumed_tokens: Dict[str, List[Token]]) -> Optional[Dict[str, List[Token]]]:
        """
        Fire the transition logic - override this method.
        
        Args:
            consumed_tokens: Dict mapping arc labels to lists of consumed tokens
            
        Returns:
            Dict mapping output arc labels to lists of tokens to produce
        """
        # Default implementation does nothing
        return {}

    async def execute(self, consumed_tokens: Dict[str, List[Token]]) -> Optional[Dict[str, List[Token]]]:
        """
        DEPRECATED: Use on_fire() instead.
        
        This method is kept for backwards compatibility.
        """
        return await self.on_fire(consumed_tokens)
    
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
        return f"{self.__class__.__name__}(fired={self.fire_count})"


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
        self._auto_build()
    
    def _auto_build(self):
        if hasattr(self, 'build'):
            self.build()
    
    def build(self):
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
    Main async Petri net execution engine.
    
    This is event-driven: transitions are checked whenever tokens are added to places.
    """
    
    def __init__(self, log_fn=print):
        self._places: Dict[Type[Place], Place] = {}
        self._transitions: List[Transition] = []
        self._io_input_places: List[IOInputPlace] = []
        self._io_output_places: List[IOOutputPlace] = []
        
        self._log_fn = log_fn or (lambda x: None)
        
        self._is_running = False
        self._is_finished = False
        self._firing_log = []
        
        # Event handling
        self._transition_check_queue = asyncio.Queue()
        self._running_tasks = set()
        
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
    
    def log(self, message: str):
        """Log a message."""
        self._log_fn(f"[{datetime.now()}] {message}")
    
    def add_interface(self, interface_class: Type[Interface]):
        """Add an interface by flattening all its components."""
        interface_instance = interface_class()
        
        for place_class in interface_instance.get_all_places():
            if place_class not in self._places:
                self.add_place(place_class)
        
        for transition_class in interface_instance.get_all_transitions():
            self.add_transition(transition_class())
        
        for nested_interface_class in interface_instance.get_all_interfaces():
            self.add_interface(nested_interface_class)
        
        return interface_instance
    
    def add_place(self, place_class: Type[Place]):
        """Add a place to the net."""
        if place_class not in self._places:
            place_instance = place_class()
            place_instance._attach_to_net(self)
            
            # Track IO places separately
            if isinstance(place_instance, IOInputPlace):
                self._io_input_places.append(place_instance)
            elif isinstance(place_instance, IOOutputPlace):
                self._io_output_places.append(place_instance)
            
            self._places[place_class] = place_instance
            self.log(f"Added place: {place_class.__name__}")
        
        return self._places[place_class]
    
    def add_transition(self, transition: Union[Transition, Type[Transition]]):
        """Add a transition to the net."""
        if isinstance(transition, type):
            transition = transition()
        
        if not isinstance(transition, Transition):
            raise InvalidTransitionType(f"{transition} is not a valid Transition")
        
        transition._attach_to_net(self)
        self._transitions.append(transition)
        self.log(f"Added transition: {transition.__class__.__name__}")
        return transition
    
    def _get_place(self, place_type: Type[Place]) -> Place:
        """Get a place by type."""
        if place_type not in self._places:
            raise KeyError(f"Place {place_type.__name__} not found")
        return self._places[place_type]
    
    async def produce_token(self, place_type: Type[Place], token: Token):
        """Produce a token to the specified place."""
        place = self._get_place(place_type)
        await place.add_token(token)
    
    async def _on_token_added(self, place: Place, token: Token):
        """Called when a token is added to any place - triggers transition checking."""
        self.log(f"Token added to {place.__class__.__name__}: {token}")
        
        # Queue transition checking
        await self._transition_check_queue.put(('token_added', place, token))
    
    async def _on_token_removed(self, place: Place, token: Token):
        """Called when a token is removed from any place."""
        self.log(f"Token removed from {place.__class__.__name__}: {token}")
    
    async def _check_and_fire_transitions(self):
        """Check all transitions and fire those that can fire."""
        # Sort transitions by priority
        sorted_transitions = sorted(self._transitions, key=lambda t: t.PRIORITY, reverse=True)
        
        for transition in sorted_transitions:
            try:
                if await transition.can_fire():
                    fired = await transition.fire()
                    if fired:
                        self.log(f"Fired transition: {transition.__class__.__name__}")
                        self._firing_log.append((datetime.now(), transition.__class__.__name__))
                        # After firing, check again (more tokens might enable more transitions)
                        await self._transition_check_queue.put(('transition_fired', transition))
            except Exception as e:
                self.log(f"Error in transition {transition.__class__.__name__}: {e}")
    
    async def _event_loop(self):
        """Main event loop that processes token additions and fires transitions."""
        while self._is_running:
            try:
                # Wait for events
                event = await asyncio.wait_for(self._transition_check_queue.get(), timeout=0.1)
                
                # Process the event
                event_type = event[0]
                if event_type in ('token_added', 'transition_fired'):
                    await self._check_and_fire_transitions()
                
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
                            self.log(f"Fired transition: {transition.__class__.__name__}")
                            self._firing_log.append((datetime.now(), transition.__class__.__name__))
                            any_fired = True
                except Exception as e:
                    self.log(f"Error in transition {transition.__class__.__name__}: {e}")
            
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
            'firing_log': [(str(time), name) for time, name in self._firing_log]
        }
    
    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram representing the Petri net structure."""
        lines = ["graph TD"]

        # Group places and transitions by interface
        interface_groups = {}
        place_to_interface = {}
        transition_to_interface = {}

        # Find all interface classes in the net
        for interface_class in getattr(self, '_discovered_interfaces', []):
            group_name = interface_class.__name__
            interface_groups[group_name] = []
            for place in interface_class._discovered_places:
                place_to_interface[place.__name__] = group_name
                interface_groups[group_name].append(place.__name__)
            for transition in interface_class._discovered_transitions:
                transition_to_interface[transition.__name__] = group_name
                interface_groups[group_name].append(transition.__name__)

        # Add places as circles (different styling for different IO types)
        for place_type, place_instance in self._places.items():
            place_name = place_type.__name__
            token_count = place_instance.token_count
            
            if isinstance(place_instance, IOInputPlace):
                # IO input places get arrow pointing in
                lines.append(f'    {place_name}(("→ {place_name}<br/>({token_count} tokens)<br/>[INPUT]"))')
            elif isinstance(place_instance, IOOutputPlace):
                # IO output places get arrow pointing out  
                lines.append(f'    {place_name}(("{place_name} →<br/>({token_count} tokens)<br/>[OUTPUT]"))')
            else:
                # Regular places
                lines.append(f'    {place_name}(("{place_name}<br/>({token_count} tokens)"))')

        # Add transitions as rectangles
        for transition in self._transitions:
            transition_name = transition.__class__.__name__
            fire_count = transition.fire_count
            lines.append(f'    {transition_name}["{transition_name}<br/>(fired {fire_count}x)"]')

        # Group nodes in subgraphs for interfaces
        for group, members in interface_groups.items():
            if members:  # Only create subgraph if it has members
                lines.append(f'    subgraph {group}')
                for member in members:
                    lines.append(f'        {member}')
                lines.append('    end')

        # Add arcs
        for transition in self._transitions:
            transition_name = transition.__class__.__name__

            # Input arcs
            input_arcs = transition._normalize_arcs(transition.input_arcs())
            for label, arc in input_arcs.items():
                place_name = arc.place.__name__
                if arc.weight > 1:
                    lines.append(f'    {place_name} -->|"{arc.weight}"| {transition_name}')
                else:
                    lines.append(f'    {place_name} --> {transition_name}')

            # Output arcs
            output_arcs = transition._normalize_arcs(transition.output_arcs())
            for label, arc in output_arcs.items():
                place_name = arc.place.__name__
                if arc.weight > 1:
                    lines.append(f'    {transition_name} -->|"{arc.weight}"| {place_name}')
                else:
                    lines.append(f'    {transition_name} --> {place_name}')

        return "\n".join(lines)
    
    def save_mermaid_diagram(self, filename: str):
        """Save the Mermaid diagram to a file."""
        with open(filename, 'w') as f:
            f.write(self.generate_mermaid_diagram())
    
    def __repr__(self):
        return f"PetriNet(places={len(self._places)}, transitions={len(self._transitions)})"


# Utility Functions
def create_simple_token(data: Any = None) -> Token:
    """Create a simple token with the given data."""
    class SimpleToken(Token):
        pass
    return SimpleToken(data)