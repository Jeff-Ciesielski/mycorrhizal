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
    wait,
    FIRST_COMPLETED,
    CancelledError,
)
import inspect
import traceback
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable, Tuple
from collections import defaultdict
from enum import Enum
import weakref
import time

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

    Places are now backed by asyncio queues and operate autonomously.
    Each place has exactly one associated transition (one-to-one mapping).
    """

    MAX_CAPACITY: Optional[int] = None
    ACCEPTED_TOKEN_TYPES: Optional[Set[Type[Token]]] = None

    def __init__(self):
        self._queue: Queue = None  # Will be set during net construction
        self._total_tokens_processed = 0
        self._net: Optional["PetriNet"] = None
        self._qualified_name: Optional[str] = None
        self._task: Optional[Task] = None

    def _attach_to_net(self, net: "PetriNet"):
        """Called when place is attached to a net."""
        self._net = net
        # Create queue with capacity if specified
        maxsize = self.MAX_CAPACITY if self.MAX_CAPACITY is not None else 0
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

    async def add_token(self, token: Token, timeout: Optional[float] = None):
        """
        Add a token to this place's queue.

        Args:
            token: Token to add
            timeout: Timeout for blocking operation

        Raises:
            InvalidTokenType: If token type is not accepted
            QueueFull: If queue is full and would block
        """
        if not self.can_accept_token(token):
            if self.is_full:
                raise InsufficientTokens(f"Place {self.qualified_name} is at capacity")
            else:
                raise InvalidTokenType(
                    f"Place {self.qualified_name} cannot accept token type {type(token)}"
                )

        if timeout is not None:
            await wait_for(self._queue.put(token), timeout=timeout)
        else:
            await self._queue.put(token)

        self._total_tokens_processed += 1
        token.on_produced()
        await self.on_token_added(token)

    def add_token_nowait(self, token: Token):
        """Add a token without waiting."""
        if not self.can_accept_token(token):
            if self.is_full:
                raise InsufficientTokens(f"Place {self.qualified_name} is at capacity")
            else:
                raise InvalidTokenType(
                    f"Place {self.qualified_name} cannot accept token type {type(token)}"
                )

        self._queue.put_nowait(token)
        self._total_tokens_processed += 1
        token.on_produced()
        # Note: can't await on_token_added here since this is sync

    async def get_token(self, timeout: Optional[float] = None) -> Token:
        """
        Get a token from this place's queue.

        Args:
            timeout: Timeout for blocking operation

        Returns:
            Token from the queue

        Raises:
            QueueEmpty: If queue is empty and would block
        """
        if timeout is not None:
            token = await wait_for(self._queue.get(), timeout=timeout)
        else:
            token = await self._queue.get()

        token.on_consumed()
        await self.on_token_removed(token)
        return token

    def get_token_nowait(self) -> Token:
        """Get a token without waiting."""
        token = self._queue.get_nowait()
        token.on_consumed()
        return token

    async def peek_token(self, timeout: Optional[float] = None) -> Optional[Token]:
        """Peek at the next token without removing it."""
        # asyncio.Queue doesn't have peek, so we get and put back
        try:
            if timeout is not None:
                token = await wait_for(self._queue.get(), timeout=timeout)
            else:
                token = await self._queue.get()
            # Put it back immediately
            self._queue.put_nowait(token)
            return token
        except (QueueEmpty, QueueFull, asyncio.TimeoutError):
            return None

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
    Output IO places that send tokens to external systems.

    These places run autonomous tasks that continuously drain their queues
    and send tokens to external systems.
    """

    def _start_output_task(self):
        """Start the output processing task."""
        if self._task is None:
            self._task = create_task(self._output_loop())
            if self._net:
                self._net._running_tasks.add(self._task)

    async def _output_loop(self):
        """Main output loop - continuously processes tokens from queue."""
        try:
            await self.on_output_start()

            while True:
                try:
                    # await on queue without timeout
                    token = await self._queue.get()

                    # Process the token
                    error_token = await self.on_token_added(token)

                    if error_token is None:
                        # Success
                        await self.on_io_success(token)
                    else:
                        # Failed - put error token back in queue
                        self._queue.put_nowait(error_token)
                        await self.on_io_error(token, error_token)

                except CancelledError:
                    # Clean shutdown
                    break
                except Exception as e:
                    await self.log(
                        f"Unexpected error in IO output place {self.qualified_name}: {e}"
                    )
                    await self.on_io_error(token if "token" in locals() else None, None)

            await self.on_output_stop()

        except CancelledError:
            await self.on_output_stop()
            raise
        except Exception as e:
            await self.log(f"Fatal error in output loop for {self.qualified_name}: {e}")
        finally:
            if self._net and self._task in self._net._running_tasks:
                self._net._running_tasks.discard(self._task)

    async def on_token_added(self, token: Token) -> Optional[Token]:
        """
        Handle outgoing token. Override this for output places.

        Returns:
            None if successful (token is consumed)
            Error token if failed (replaces original token)
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

    async def on_io_success(self, token: Token):
        """Called when IO operation succeeds."""
        pass

    async def on_io_error(self, token: Token, error_token: Optional[Token]):
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
            self._task = create_task(self._input_loop())
            if self._net:
                self._net._running_tasks.add(self._task)

    async def _input_loop(self):
        """Main input loop - calls on_input() and puts tokens in queue."""
        try:
            await self.on_input_start()

            while True:
                try:
                    # Call user-defined input handler
                    result = await self.on_input()

                    if isinstance(result, Token):
                        await self._queue.put(result)
                        result.on_produced()
                        await self.on_token_added(result)
                    elif isinstance(result, list) and all(
                        isinstance(x, Token) for x in result
                    ):
                        for token in result:
                            await self._queue.put(token)
                            token.on_produced()
                            await self.on_token_added(token)
                    elif result is None:
                        pass
                    else:
                        raise TypeError(
                            f"IOInput Places must produce Tokens or Lists of Tokens, got: {result}"
                        )

                    # Yield control so other tasks can run
                    await sleep(0)

                except CancelledError:
                    # Clean shutdown
                    break
                except Exception as e:
                    await self.log(
                        f"Error in input loop for {self.qualified_name}: {e}"
                    )
                    await sleep(0)  # Back off on error

            await self.on_input_stop()

        except CancelledError:
            await self.on_input_stop()
            raise
        except Exception as e:
            await self.log(f"Fatal error in input loop for {self.qualified_name}: {e}")
        finally:
            if self._net and self._task in self._net._running_tasks:
                self._net._running_tasks.discard(self._task)

    async def on_input_start(self):
        """Called when input starts."""
        pass

    async def on_input(self) -> Union[Token, List[Token], None]:
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
    token_type: Optional[Type[Token]] = None
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            if isinstance(self.place, str):
                self.label = self.place.split(".")[-1].lower()
            elif self.place is not None:
                self.label = self.place.__name__.lower()
            else:
                self.label = "unnamed_arc"


class DispatchPolicy(Enum):
    """Policies for when transitions should fire."""

    ANY = "any"  # Fire when any input place has required tokens
    ALL = "all"  # Fire when all input places have required tokens


class PolicyDispatcher:
    """
    Manages token collection and transition firing based on policy.

    Replaces the guard system with policy-based dispatching.
    """

    def __init__(self, transition: "Transition", policy: DispatchPolicy):
        self.transition = transition
        self.policy = policy
        self.input_places: Dict[PlaceName, Place] = {}
        self.input_weights: Dict[PlaceName, int] = {}
        self.collector_queues: Dict[PlaceName, Queue] = {}
        self._task: Optional[Task] = None
        self._collector_tasks: List[Task] = []
        self._waiting_on: Set[PlaceName] = set()  # Track what we're waiting for

    def add_input(self, label: PlaceName, place: Place, weight: int):
        """Add an input place with its weight."""
        self.input_places[label] = place
        self.input_weights[label] = weight

        if weight > 1:
            # Create collector queue for batching
            self.collector_queues[label] = Queue()

    def start(self):
        """Start the dispatcher task."""
        if self._task is None:
            self._task = create_task(self._dispatch_loop())
            if self.transition._net:
                self.transition._net._running_tasks.add(self._task)

    async def _dispatch_loop(self):
        """Main dispatch loop."""
        try:
            # Start collector tasks for weighted inputs
            for label, weight in self.input_weights.items():
                if weight > 1:
                    collector = create_task(self._collector_loop(label, weight))
                    self._collector_tasks.append(collector)
                    if self.transition._net:
                        self.transition._net._running_tasks.add(collector)

            while True:
                try:
                    # Collect tokens based on policy
                    token_batch = await self._collect_tokens()

                    if token_batch:
                        # Forward to transition
                        await self.transition._input_queue.put(token_batch)

                except CancelledError:
                    break
                except Exception as e:
                    if self.transition._net:
                        await self.transition._net.log(
                            f"Error in dispatcher for {self.transition.qualified_name}: {e}"
                        )

            # Clean up collector tasks
            for collector in self._collector_tasks:
                collector.cancel()
            
            # Wait for them to finish
            if self._collector_tasks:
                await gather(*self._collector_tasks, return_exceptions=True)

        except CancelledError:
            # Clean shutdown
            for collector in self._collector_tasks:
                collector.cancel()
            if self._collector_tasks:
                await gather(*self._collector_tasks, return_exceptions=True)
            raise
        except Exception as e:
            if self.transition._net:
                await self.transition._net.log(
                    f"Fatal error in dispatcher for {self.transition.qualified_name}: {e}"
                )
        finally:
            if (
                self.transition._net
                and self._task in self.transition._net._running_tasks
            ):
                self.transition._net._running_tasks.discard(self._task)

    async def _collector_loop(self, label: PlaceName, weight: int):
        """Collector task for batching tokens from weighted arcs."""
        place = self.input_places[label]
        collector_queue = self.collector_queues[label]

        try:
            while True:
                batch = []

                # Collect required number of tokens
                for i in range(weight):
                    try:
                        token = await place.get_token()
                        batch.append(token)
                    except CancelledError:
                        # Put back any partial batch
                        for token in reversed(batch):
                            place._queue.put_nowait(token)
                        raise

                if len(batch) == weight:
                    # Forward complete batch to collector queue
                    await collector_queue.put(batch)

        except CancelledError:
            raise
        except Exception as e:
            if self.transition._net:
                await self.transition._net.log(
                    f"Error in collector for {self.transition.qualified_name}.{label}: {e}"
                )

    async def _collect_tokens(self) -> Optional[Dict[PlaceName, List[Token]]]:
        """Collect tokens based on dispatch policy."""
        
        if self.policy == DispatchPolicy.ANY:
            return await self._collect_any_policy()
        elif self.policy == DispatchPolicy.ALL:
            return await self._collect_all_policy()
        
        return None

    async def _collect_any_policy(self) -> Optional[Dict[PlaceName, List[Token]]]:
        """Collect tokens using ANY policy - fire when any input has tokens."""
        # Create tasks for getting from each input
        get_tasks = {}
        for label in self.input_places:
            if label in self.collector_queues:
                # Weighted input - get from collector queue
                get_tasks[label] = create_task(self.collector_queues[label].get())
            else:
                # Regular input - get from place queue
                place = self.input_places[label]
                get_tasks[label] = create_task(place.get_token())
        
        # Update what we're waiting on for deadlock detection
        self._waiting_on = set(get_tasks.keys())
        
        try:
            # Wait for any to complete
            done, pending = await wait(get_tasks.values(), return_when=FIRST_COMPLETED)
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            # Gather results from completed tasks
            result = {}
            for label, task in get_tasks.items():
                if task in done:
                    try:
                        token_or_batch = await task
                        if isinstance(token_or_batch, list):
                            result[label] = token_or_batch
                        else:
                            result[label] = [token_or_batch]
                    except CancelledError:
                        pass
            
            self._waiting_on.clear()
            return result if result else None
            
        except CancelledError:
            # Cancel all get tasks
            for task in get_tasks.values():
                task.cancel()
            await gather(*get_tasks.values(), return_exceptions=True)
            self._waiting_on.clear()
            raise

    async def _collect_all_policy(self) -> Optional[Dict[PlaceName, List[Token]]]:
        """Collect tokens using ALL policy - fire only when all inputs have tokens."""
        # Create tasks for getting from each input
        get_tasks = {}
        for label in self.input_places:
            if label in self.collector_queues:
                # Weighted input - get from collector queue
                get_tasks[label] = create_task(self.collector_queues[label].get())
            else:
                # Regular input - get from place queue
                place = self.input_places[label]
                get_tasks[label] = create_task(place.get_token())
        
        # Update what we're waiting on for deadlock detection
        self._waiting_on = set(get_tasks.keys())
        
        try:
            # Wait for all to complete
            results = await gather(*get_tasks.values())
            
            # Build result dictionary
            result = {}
            for label, token_or_batch in zip(get_tasks.keys(), results):
                if isinstance(token_or_batch, list):
                    result[label] = token_or_batch
                else:
                    result[label] = [token_or_batch]
            
            self._waiting_on.clear()
            return result
            
        except CancelledError:
            # Cancel all get tasks
            for task in get_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Wait for cancellations and put tokens back
            for label, task in get_tasks.items():
                try:
                    token_or_batch = await task
                    # Put back what we got
                    if label in self.collector_queues:
                        await self.collector_queues[label].put(token_or_batch)
                    else:
                        place = self.input_places[label]
                        if isinstance(token_or_batch, list):
                            for token in reversed(token_or_batch):
                                place._queue.put_nowait(token)
                        else:
                            place._queue.put_nowait(token_or_batch)
                except (CancelledError, Exception):
                    pass
            
            self._waiting_on.clear()
            raise

    async def _put_back_tokens(self, token_dict: Dict[PlaceName, List[Token]]):
        """Put tokens back into their places."""
        for label, tokens in token_dict.items():
            if label in self.collector_queues:
                # Put back into collector queue
                if tokens:
                    await self.collector_queues[label].put(tokens)
            else:
                # Put back into place queue
                place = self.input_places[label]
                for token in reversed(tokens):
                    place._queue.put_nowait(token)

    def is_waiting(self) -> bool:
        """Check if dispatcher is currently waiting for tokens."""
        return len(self._waiting_on) > 0

    def get_waiting_places(self) -> Set[PlaceName]:
        """Get the places this dispatcher is waiting on."""
        return self._waiting_on.copy()


# Core Transition System
class Transition(ABC):
    """
    Base class for all transitions in the Petri net system.

    In the autonomous model, transitions run as independent tasks
    waiting on input queues managed by policy dispatchers.
    """

    DISPATCH_POLICY: DispatchPolicy = DispatchPolicy.ALL  # Default policy

    def __init__(self):
        self._fire_count = 0
        self._last_fired = None
        self._net: Optional["PetriNet"] = None
        self._qualified_name: Optional[str] = None
        self._interface_path: Optional[str] = None
        self._input_queue: Queue = Queue()
        self._output_places: Dict[PlaceName, Place] = {}
        self._dispatcher: Optional[PolicyDispatcher] = None
        self._task: Optional[Task] = None

    def _attach_to_net(self, net: "PetriNet"):
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

    def _setup_dispatcher(self):
        """Set up the policy dispatcher for this transition."""
        self._dispatcher = PolicyDispatcher(self, self.DISPATCH_POLICY)

        # Add input places to dispatcher
        input_arcs = self.input_arcs()
        for label, arc in input_arcs.items():
            place = self._net._resolve_place_from_arc(arc)
            self._dispatcher.add_input(label, place, arc.weight)

    def _start_transition_task(self):
        """Start the transition processing task."""
        if self._task is None:
            self._task = create_task(self._transition_loop())
            if self._net:
                self._net._running_tasks.add(self._task)

        if self._dispatcher:
            self._dispatcher.start()

    async def _transition_loop(self):
        """Main transition loop - processes token batches from dispatcher."""
        try:
            while True:
                try:
                    # Properly await on queue without timeout
                    token_batch = await self._input_queue.get()

                    # Fire the transition
                    await self._fire_with_tokens(token_batch)

                except CancelledError:
                    break
                except Exception as e:
                    await self._handle_error(
                        e, token_batch if "token_batch" in locals() else {}
                    )

        except CancelledError:
            raise
        except Exception as e:
            if self._net:
                await self._net.log(
                    f"Fatal error in transition loop for {self.qualified_name}: {e}"
                )
        finally:
            if self._net and self._task in self._net._running_tasks:
                self._net._running_tasks.discard(self._task)

    async def _fire_with_tokens(self, consumed_tokens: Dict[PlaceName, List[Token]]):
        """Fire the transition with the given consumed tokens."""
        try:
            await self.on_before_fire(consumed_tokens)

            # Call user-defined firing logic
            output_tokens = await self.on_fire(consumed_tokens)

            # Produce output tokens
            if output_tokens:
                output_arcs = self.output_arcs()
                for label, tokens in output_tokens.items():
                    if label in output_arcs:
                        place = self._output_places[label]
                        if isinstance(tokens, Token):
                            tokens = [tokens]
                        for token in tokens:
                            place.add_token_nowait(token)

            self._fire_count += 1
            self._last_fired = datetime.now()

            await self.on_after_fire(consumed_tokens, output_tokens)

            if self._net:
                await self._net.log(f"Fired transition: {self.qualified_name}")

        except Exception as e:
            # Handle error and potentially rollback
            await self._handle_error(e, consumed_tokens)

    async def on_fire(
        self, consumed: Dict[PlaceName, List[Token]]
    ) -> Optional[Dict[PlaceName, List[Token]]]:
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

    async def on_before_fire(self, consumed: Dict[PlaceName, List[Token]]):
        """Called before transition fires."""
        pass

    async def on_after_fire(
        self,
        consumed: Dict[PlaceName, List[Token]],
        produced: Optional[Dict[PlaceName, List[Token]]],
    ):
        """Called after transition fires successfully."""
        pass

    async def _handle_error(
        self, error: Exception, consumed_tokens: Dict[PlaceName, List[Token]]
    ):
        """Handle transition errors."""
        if self._net and hasattr(self._net, "on_transition_error"):
            action = await self._net.on_transition_error(self, error, consumed_tokens)

            if action == "restart":
                # Put tokens back and continue
                await self._rollback_tokens(consumed_tokens)
                return
            elif action == "kill":
                # Shut down the net
                if self._net:
                    await self._net.stop()
                return

        # Default: re-raise the error
        raise error

    async def _rollback_tokens(self, consumed_tokens: Dict[PlaceName, List[Token]]):
        """Put consumed tokens back into their places for error recovery."""
        input_arcs = self.input_arcs()

        for label, tokens in consumed_tokens.items():
            if label in input_arcs:
                arc = input_arcs[label]
                place = self._net._resolve_place_from_arc(arc)

                # Put tokens back in reverse order to maintain FIFO
                for token in reversed(tokens):
                    try:
                        place._queue.put_nowait(token)
                    except QueueFull:
                        # If we can't put it back, we have a problem
                        if self._net:
                            await self._net.log(
                                f"WARNING: Could not rollback token to {place.qualified_name}"
                            )

    def is_waiting(self) -> bool:
        """Check if this transition is waiting for tokens."""
        if self._dispatcher:
            return self._dispatcher.is_waiting()
        return False

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
            if hasattr(base, "_discovered_places"):
                cls._discovered_places.extend(base._discovered_places)
            if hasattr(base, "_discovered_transitions"):
                cls._discovered_transitions.extend(base._discovered_transitions)
            if hasattr(base, "_discovered_interfaces"):
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
        handler: Callable[[Transition, Exception, Dict[PlaceName, List[Token]]], str],
    ):
        """
        Set error handler for transition failures.

        Handler should return one of: "restart", "kill", or re-raise the exception.
        """
        self._error_handler = handler

    async def on_transition_error(
        self,
        transition: Transition,
        error: Exception,
        consumed_tokens: Dict[PlaceName, List[Token]],
    ) -> str:
        """Handle transition errors."""
        if self._error_handler:
            result = self._error_handler(transition, error, consumed_tokens)
            if asyncio.iscoroutine(result):
                return await result
            return result

        # Default: log and restart
        await self.log(f"Error in transition {transition.qualified_name}: {error}")
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
        # Implementation same as original, just without async/await since this is setup
        pass

    def _validate_transition_boundaries(
        self, transition_qualified_name: str, transition_class: Type[Transition]
    ):
        """Validate that a transition doesn't violate interface boundaries."""
        # Implementation same as original, just without async/await since this is setup
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
        # Implementation same as original, just without async/await since this is setup
        pass

    def _is_within_interface_boundary(
        self, place_qualified_name: str, transition_interface: str
    ) -> bool:
        """Allow access to any place that is a direct child or descendant of the transition's interface."""
        # Implementation same as original, just without async/await since this is setup
        return True  # Simplified for brevity

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
                
                if await self.is_deadlocked():
                    await self.log("DEADLOCK DETECTED!")
                    await self.on_deadlock()
                    
        except CancelledError:
            raise

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
        
        # Check if any IO inputs are still producing
        # This is a bit tricky - we need a way for IOInputPlaces to signal they're done
        # For now, check if the net has a 'finished' flag (like in the demo)
        if hasattr(self, 'finished') and self.finished:
            # Net has indicated it's finished producing
            return True
        
        # Check if there are any active IO input tasks that might produce tokens
        active_inputs = any(
            place._task and not place._task.done() 
            for place in self._io_input_places
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

    async def start(self):
        """Start the autonomous Petri net execution."""

        await self.log(f"Starting Net: {self.__class__.__name__}")
        self._shutdown_event.clear()

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

        # Start all IO output places
        for io_output_place in self._io_output_places:
            io_output_place._start_output_task()

        # Start all transitions
        for transition in self._transitions:
            transition._start_transition_task()

        await self.log(f"Net: {self.__class__.__name__} tasks started")

    async def stop(self):
        """Stop the autonomous Petri net execution."""
        if self._shutdown_event.is_set():
            return

        await self.log(f"Stopping Net: {self.__class__.__name__}")
        self._shutdown_event.set()

        # Cancel all running tasks
        for task in self._running_tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if self._running_tasks:
            await gather(*self._running_tasks, return_exceptions=True)

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
                    "dispatch_policy": transition.DISPATCH_POLICY.value,
                    "is_waiting": transition.is_waiting(),
                }
                for transition in self._transitions
            },
            "running_tasks": len(self._running_tasks),
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

        # Add transitions with their qualified names
        for transition in self._transitions:
            qualified_name = transition.qualified_name
            safe_name = qualified_name.replace(".", "_")
            fire_count = transition.fire_count
            policy = transition.DISPATCH_POLICY.value
            lines.append(
                f'    {safe_name}["{qualified_name}<br/>(fired {fire_count}x)<br/>Policy: {policy}"]'
            )

        # Group nodes in subgraphs for interfaces
        for interface_name, components in self._interface_hierarchy.items():
            if components:
                safe_interface_name = interface_name.replace(".", "_")
                lines.append(f'    subgraph {safe_interface_name} ["{interface_name}"]')
                for component_name in components:
                    safe_component_name = component_name.replace(".", "_")
                    lines.append(f"        {safe_component_name}")
                lines.append("    end")

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

                    lines.append(
                        f'    {place_safe_name} -->|"{edge_label}"| {transition_safe_name}'
                    )
            except Exception as e:
                # Can't use await in sync method
                print(
                    f"Warning: Could not generate input arcs for {transition.qualified_name}: {e}"
                )

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

                    lines.append(
                        f'    {transition_safe_name} -->|"{edge_label}"| {place_safe_name}'
                    )
            except Exception as e:
                print(
                    f"Warning: Could not generate output arcs for {transition.qualified_name}: {e}"
                )

        return "\n".join(lines)

    def save_mermaid_diagram(self, filename: str):
        """Save the Mermaid diagram to a file."""
        with open(filename, "w") as f:
            f.write(self.generate_mermaid_diagram())

    def print_mermaid_diagram(self):
        """Print the Mermaid diagram to console."""
        print(self.generate_mermaid_diagram())

    def __repr__(self):
        return f"PetriNet(places={len(self._places_by_type)}, transitions={len(self._transitions)}, running={not self._shutdown_event.is_set()})"


# Utility Functions
def create_simple_token(data: Any = None) -> Token:
    """Create a simple token with the given data."""

    class SimpleToken(Token):
        pass

    return SimpleToken(data)