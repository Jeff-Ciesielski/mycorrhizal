#!/usr/bin/env python3
"""
Hypha DSL - Runtime Layer

Runtime objects that execute the Petri net specification.
Manages token flow, transition firing, and asyncio task coordination.
"""

import asyncio
from asyncio import Event, Task
from typing import Any, List, Dict, Optional, Set, Tuple, Callable, Deque, Union, Type, get_type_hints
from collections import deque, OrderedDict
from itertools import product, combinations
import inspect
import logging

from mycorrhizal.common.wrappers import create_view_from_protocol
from mycorrhizal.common.compilation import (
    _get_compiled_metadata,
    _clear_compilation_cache,
    CompiledMetadata,
)
from mycorrhizal.common.cache import InterfaceViewCache
from .specs import NetSpec, PlaceSpec, TransitionSpec, ArcSpec, PlaceType, GuardSpec

logger = logging.getLogger(__name__)


try:
    # Library should not configure root logging; be quiet by default
    logger.addHandler(logging.NullHandler())
except Exception:
    # NullHandler may not be available in very old Pythons; ignore if so
    pass
# Library does not configure handlers by default. Callers may configure logging.


class PlaceRuntime:
    """Runtime execution of a place"""

    def __init__(self, spec: PlaceSpec, bb: Any, timebase: Any, fqn: str, net: Optional['NetRuntime'] = None):
        self.spec = spec
        self.fqn = fqn
        self.bb = bb
        self.timebase = timebase
        self.net = net  # Reference to NetRuntime for cache access
        self.state = spec.state_factory() if spec.state_factory else None
        
        # tokens can be a bag (list) or queue (deque)
        self.tokens: Union[List[Any], Deque[Any]]
        if spec.place_type == PlaceType.BAG:
            self.tokens = []
        else:  # QUEUE
            self.tokens = deque()
        
        self.token_added_event = Event()
        self.token_removed_event = Event()
        
        self.io_task: Optional[Task] = None
    
    def add_token(self, token: Any):
        """Add a token to this place"""
        if self.spec.place_type == PlaceType.BAG:
            self.tokens.append(token)
        else:  # QUEUE
            self.tokens.append(token)
        
        self.token_added_event.set()
    
    def remove_tokens(self, tokens_to_remove: List[Any]):
        """Remove specific tokens from this place.

        Note: This operation is idempotent - if a token has already been removed
        (e.g., by a competing transition), it will be safely skipped.
        """
        if self.spec.place_type == PlaceType.BAG:
            for token in tokens_to_remove:
                try:
                    self.tokens.remove(token)
                except ValueError:
                    # Token already removed by another transition (race condition)
                    # This is expected when multiple transitions compete
                    pass
        else:  # QUEUE
            # Try to use set for O(1) lookup if tokens are hashable
            try:
                removed_set = set(tokens_to_remove)
                temp = deque()
                for token in self.tokens:
                    if token not in removed_set:
                        temp.append(token)
                self.tokens = temp
            except TypeError:
                # Fall back to O(n) lookup for unhashable tokens (e.g., Task objects)
                temp = deque()
                for token in self.tokens:
                    if token not in tokens_to_remove:
                        temp.append(token)
                self.tokens = temp

        self.token_removed_event.set()
        self.token_removed_event.clear()
    
    def peek_tokens(self, count: int) -> List[Any]:
        """Peek at tokens without removing them"""
        if len(self.tokens) < count:
            return []
        # Normalize to a list slice for both BAG and QUEUE
        return list(self.tokens)[:count]
    
    async def start_io_input(self):
        """Start IOInputPlace generator task"""
        if not self.spec.is_io_input or not self.spec.handler:
            return

        # Create interface view if handler has interface type hint
        if self.net:
            bb_to_pass = self.net._create_interface_view_if_needed(self.bb, self.spec.handler)
        else:
            bb_to_pass = self.bb

        async def io_input_loop():
            try:
                handler = self.spec.handler
                if handler is None:
                    return

                sig = inspect.signature(handler)
                if len(sig.parameters) == 2:
                    gen = handler(bb_to_pass, self.timebase)
                else:
                    gen = handler()  # type: ignore[call-arg]

                async for token in gen:
                    self.add_token(token)
            except asyncio.CancelledError:
                pass

        self.io_task = asyncio.create_task(io_input_loop())
    
    async def handle_io_output(self, token: Any):
        """Handle IOOutputPlace token"""
        if not self.spec.is_io_output or not self.spec.handler:
            return

        # Create interface view if handler has interface type hint
        if self.net:
            bb_to_pass = self.net._create_interface_view_if_needed(self.bb, self.spec.handler)
        else:
            bb_to_pass = self.bb

        sig = inspect.signature(self.spec.handler)
        if len(sig.parameters) == 3:
            await self.spec.handler(token, bb_to_pass, self.timebase)
        else:
            await self.spec.handler(token)
    
    async def cancel(self):
        """Cancel any running IO tasks"""
        if self.io_task:
            self.io_task.cancel()
            try:
                await self.io_task
            except asyncio.CancelledError:
                pass


class TransitionRuntime:
    """Runtime execution of a transition"""
    
    def __init__(self, spec: TransitionSpec, net: 'NetRuntime', fqn: str):
        self.spec = spec
        self.fqn = fqn
        self.net = net
        self.state = spec.state_factory() if spec.state_factory else None
        # input_arcs now use canonical parts tuples as keys
        self.input_arcs: List[Tuple[Tuple[str, ...], ArcSpec]] = []
        self.output_arcs: List[ArcSpec] = []

        self.task: Optional[Task] = None
        self._stop_event = Event()
        self._spurious_wakeup_count = 0
    
    def add_input_arc(self, place_parts: Tuple[str, ...], arc: ArcSpec):
        """Register an input arc"""
        self.input_arcs.append((place_parts, arc))
    
    def add_output_arc(self, arc: ArcSpec):
        """Register an output arc"""
        self.output_arcs.append(arc)
    
    def _generate_token_combinations(self):
        """
        Generate all valid token combinations for input arcs.
        Returns generator of tuples, where each tuple contains sub-tuples for each arc.
        Using generator instead of list for lazy evaluation - stops after guard selects first valid combination.
        """
        arc_tokens = []

        for place_parts, arc in self.input_arcs:
            place = self.net.places[place_parts]
            tokens = place.peek_tokens(arc.weight)

            if len(tokens) < arc.weight:
                return iter([])  # Empty generator if insufficient tokens

            if arc.weight == 1:
                arc_tokens.append([(t,) for t in tokens])
            else:
                arc_tokens.append(list(combinations(tokens, arc.weight)))

        if not arc_tokens:
            return iter([])  # Empty generator

        # Return generator instead of list - lazy evaluation
        return product(*arc_tokens)

    async def _check_guard(self, combinations_generator) -> Optional[Tuple[Tuple[Any, ...], ...]]:
        """Check guard and return combination to consume, or None"""
        # If no guard, return first combination from generator
        if not self.spec.guard:
            try:
                return next(iter(combinations_generator))
            except StopIteration:
                return None

        # Guard exists - need to build list for guard evaluation
        # Guards expect to see all combinations to choose from
        combinations_list = list(combinations_generator)

        if not combinations_list:
            return None

        guard_func = self.spec.guard.func
        guard_result = guard_func(combinations_list, self.net.bb, self.net.timebase)

        if inspect.isgenerator(guard_result):
            for result in guard_result:
                if result is not None:
                    return result
        elif inspect.isasyncgen(guard_result):
            async for result in guard_result:
                if result is not None:
                    return result

        return None
    
    async def _fire_transition(self, consumed_combination: Tuple[Tuple[Any, ...], ...]):
        """Execute the transition with consumed tokens"""
        consumed_flat = []
        tokens_to_remove = {}

        for i, (place_parts, arc) in enumerate(self.input_arcs):
            arc_tokens = consumed_combination[i]
            consumed_flat.extend(arc_tokens)

            if place_parts not in tokens_to_remove:
                tokens_to_remove[place_parts] = []
            tokens_to_remove[place_parts].extend(arc_tokens)

        # Remove tokens from input places
        for place_parts, tokens in tokens_to_remove.items():
            self.net.places[place_parts].remove_tokens(tokens)

        # Create interface view if handler has interface type hint
        bb_to_pass = self.net._create_interface_view_if_needed(self.net.bb, self.spec.handler)

        sig = inspect.signature(self.spec.handler)
        param_count = len(sig.parameters)

        if self.state is not None:
            logger.debug("[fire] %s consumed=%s", self.fqn, consumed_flat)
            results = self.spec.handler(consumed_flat, bb_to_pass, self.net.timebase, self.state)
        else:
            logger.debug("[fire] %s consumed=%s", self.fqn, consumed_flat)
            results = self.spec.handler(consumed_flat, bb_to_pass, self.net.timebase)
        logger.debug("[fire] results_type=%s isasyncgen=%s iscoroutine=%s",
                     type(results), inspect.isasyncgen(results), inspect.iscoroutine(results))

        if inspect.isasyncgen(results):
            async for yielded in results:
                await self._process_yield(yielded)
        elif inspect.iscoroutine(results):
            result = await results
            if result is not None:
                await self._process_yield(result)

    def _resolve_place_ref(self, place_ref) -> Optional[Tuple[str, ...]]:
        """Resolve a PlaceRef to runtime place parts.

        Attempts multiple resolution strategies in order:
        1. Exact match using PlaceRef.get_parts()
        2. Parent-relative mapping (for subnet instances)
        3. Suffix fallback (match by local name)

        Returns:
            Tuple of place parts if found, None otherwise
        """
        # Try to get parts from the PlaceRef API
        try:
            parts = tuple(place_ref.get_parts())
        except Exception:
            parts = None

        logger.debug("[resolve] place_ref=%r parts=%s", place_ref, parts)

        if parts:
            key = tuple(parts)
            if key in self.net.places:
                logger.debug("[resolve] exact match parts=%s", parts)
                return key

        # Map relative to this transition's parent parts
        trans_parts = tuple(self.fqn.split('.'))
        if len(trans_parts) > 1:
            parent_prefix = trans_parts[:-1]
            candidate = tuple(list(parent_prefix) + [place_ref.local_name])
            if candidate in self.net.places:
                logger.debug("[resolve] mapped %s -> %s using parent_prefix", parts, candidate)
                return candidate

        # Fallback: find any place whose last segment matches local_name
        for p in self.net.places.keys():
            if isinstance(p, tuple):
                segs = list(p)
            else:
                segs = p.split('.')
            if segs and segs[-1] == place_ref.local_name:
                logger.debug("[resolve] suffix match %s -> %s", place_ref.local_name, p)
                return tuple(segs)

        return None

    def _normalize_to_parts(self, key) -> Tuple[str, ...]:
        """Normalize a place key to tuple of parts for runtime lookup.

        Handles various input formats:
        - tuple: returned as-is
        - list: converted to tuple
        - str: split on '.' and converted to tuple
        - other: stringified and split on '.'
        """
        if isinstance(key, tuple):
            return key
        if isinstance(key, list):
            return tuple(key)
        if isinstance(key, str):
            return tuple(key.split('.'))
        # unknown type; try to stringify
        return tuple(str(key).split('.'))

    async def _add_token_to_place(self, place_key: Tuple[str, ...], token: Any):
        """Add a token to the specified place, handling IO output places.

        Args:
            place_key: Tuple of parts identifying the place
            token: Token to add
        """
        if place_key in self.net.places:
            place = self.net.places[place_key]
            if place.spec.is_io_output:
                logger.debug("[process_yield] calling io_output on %s token=%r", place_key, token)
                await place.handle_io_output(token)
            else:
                logger.debug("[process_yield] adding token to %s token=%r", place_key, token)
                place.add_token(token)

    async def _process_yield(self, yielded):
        """Process yielded output from transition"""
        if isinstance(yielded, dict):
            await self._process_dict_yield(yielded)
        else:
            await self._process_single_yield(yielded)

    async def _process_dict_yield(self, yielded: dict):
        """Process dictionary with place keys and token values.

        Supports:
        - Explicit place -> token mappings
        - Wildcard ('*') token that goes to all output places
        """
        wildcard_token = yielded.get('*')
        explicit_targets = set()

        for key, token in yielded.items():
            if key == '*':
                continue

            place_parts = self._resolve_place_ref(key)
            if place_parts is None:
                continue

            key_normalized = self._normalize_to_parts(place_parts)
            explicit_targets.add(key_normalized)
            await self._add_token_to_place(key_normalized, token)

        if wildcard_token is not None:
            await self._expand_wildcard_to_outputs(wildcard_token, explicit_targets)

    async def _expand_wildcard_to_outputs(self, wildcard_token: Any, explicit_targets: set):
        """Expand a wildcard token to all output places not explicitly targeted.

        Args:
            wildcard_token: Token to distribute to all output places
            explicit_targets: Set of place keys already explicitly targeted
        """
        for arc in self.output_arcs:
            # normalize target to parts
            try:
                target_parts = tuple(arc.target.get_parts())
                target_key = target_parts
            except Exception:
                # fallback if target is a string
                target_key = self._normalize_to_parts(arc.target)

            if target_key not in explicit_targets:
                await self._add_token_to_place(target_key, wildcard_token)

    async def _process_single_yield(self, yielded):
        """Process a single (place_ref, token) tuple yield."""
        place_ref, token = yielded
        place_parts = self._resolve_place_ref(place_ref)

        if place_parts is None:
            return

        key = self._normalize_to_parts(place_parts)
        await self._add_token_to_place(key, token)
    
    async def run(self):
        """Main transition execution loop"""
        try:
            while not self._stop_event.is_set():
                # Generator transitions (no inputs) fire continuously
                if not self.input_arcs:
                    # No input places - this is a generator transition
                    # Fire continuously with empty consumed list
                    logger.debug("[trans] %s is a generator, firing continuously", self.fqn)
                    try:
                        await self._fire_transition([])
                    except Exception as e:
                        logger.exception("[%s] Generator transition error", self.fqn)

                    # Small delay to prevent busy-waiting
                    await asyncio.sleep(0.01)
                    continue

                wait_tasks = []
                for place_fqn, arc in self.input_arcs:
                    place = self.net.places[place_fqn]
                    wait_tasks.append(asyncio.create_task(place.token_added_event.wait()))

                if not wait_tasks:
                    break

                stop_task = asyncio.create_task(self._stop_event.wait())
                wait_tasks.append(stop_task)

                done, pending = await asyncio.wait(
                    wait_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for task in pending:
                    task.cancel()

                if self._stop_event.is_set():
                    break
                # Debug: indicate transition woke
                logger.debug("[trans] %s woke; checking inputs", self.fqn)

                combinations = self._generate_token_combinations()

                # Check if there are any combinations (for generator, peek at first element)
                combinations_iter = iter(combinations)
                try:
                    first_combo = next(combinations_iter)
                    has_combinations = True
                    # Put it back by creating a new generator with first element
                    # We need to materialize the first element and chain it with the rest
                    def chain_with_first(first, rest_iter):
                        yield first
                        yield from rest_iter
                    combinations = chain_with_first(first_combo, combinations_iter)
                except StopIteration:
                    has_combinations = False

                logger.debug("[trans] %s has_combinations=%s", self.fqn, has_combinations)

                if has_combinations:
                    to_consume = await self._check_guard(combinations)
                    if to_consume:
                        await self._fire_transition(to_consume)

                        # Check if there are still tokens to process after firing
                        remaining_combinations = self._generate_token_combinations()
                        remaining_iter = iter(remaining_combinations)
                        try:
                            next(remaining_iter)
                            has_remaining = True
                        except StopIteration:
                            has_remaining = False

                        if not has_remaining:
                            # No more tokens, clear the event and wait for new tokens
                            for place_fqn, arc in self.input_arcs:
                                place = self.net.places[place_fqn]
                                place.token_added_event.clear()
                            # else: there are still tokens, loop again immediately
                else:
                    # No combinations available (event was set spuriously or tokens consumed by another transition)
                    # Spurious wakeup - clear events and go back to waiting
                    self._spurious_wakeup_count += 1
                    logger.debug(
                        "[trans] %s spurious wakeup #%d, clearing events",
                        self.fqn, self._spurious_wakeup_count
                    )
                    for place_fqn, arc in self.input_arcs:
                        place = self.net.places[place_fqn]
                        place.token_added_event.clear()

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("[%s] Transition error", self.fqn)
    
    async def cancel(self):
        """Cancel this transition's task"""
        self._stop_event.set()
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    def get_spurious_wakeup_count(self) -> int:
        """Get the number of spurious wakeups for this transition.

        Useful for monitoring and debugging race conditions.

        Returns:
            Number of spurious wakeups since transition started
        """
        return self._spurious_wakeup_count


class NetRuntime:
    """Runtime execution of a complete Petri net"""

    def __init__(self, spec: NetSpec, bb: Any, timebase: Any):
        self.spec = spec
        self.bb = bb
        self.timebase = timebase
        # Use tuple-of-parts as canonical keys internally
        # Keep as Any to simplify gradual migration (place/transition runtime objects)
        self.places: Dict[Tuple[str, ...], Any] = {}
        self.transitions: Dict[Tuple[str, ...], Any] = {}
        # Instance-local cache for interface views (no global state)
        self._interface_cache = InterfaceViewCache(maxsize=256)

        self._flatten_spec(spec)
        self._build_runtime()
    
    def _flatten_spec(self, spec: NetSpec):
        """Flatten nested subnet specs into flat dictionaries by computing FQNs"""
        # Register all places with their computed FQNs
        for place_name, place_spec in spec.places.items():
            place_parts = tuple(spec.get_parts(place_name))
            self.places[place_parts] = None
        
        # Register all transitions with their computed FQNs
        for trans_name, trans_spec in spec.transitions.items():
            trans_parts = tuple(spec.get_parts(trans_name))
            self.transitions[trans_parts] = None
        
        # Recursively flatten subnets
        for subnet_name, subnet_spec in spec.subnets.items():
            self._flatten_spec(subnet_spec)
    
    def _build_runtime(self):
        """Build runtime objects from flattened spec"""
        # Build all place runtimes
        for place_parts in list(self.places.keys()):
            place_spec = self._find_place_spec_by_parts(place_parts)
            self.places[place_parts] = PlaceRuntime(place_spec, self.bb, self.timebase, '.'.join(place_parts), self)
        
        # Build all transition runtimes  
        for trans_parts in list(self.transitions.keys()):
            trans_spec = self._find_transition_spec_by_parts(trans_parts)
            self.transitions[trans_parts] = TransitionRuntime(trans_spec, self, '.'.join(trans_parts))
        
        # Connect arcs by resolving references
        for arc in self._collect_all_arcs():
            # Use canonical tuple parts computed by ArcSpec
            source_parts = tuple(arc.source_parts)
            target_parts = tuple(arc.target_parts)

            if target_parts in self.transitions:
                trans = self.transitions[target_parts]
                trans.add_input_arc(source_parts, arc)

            if source_parts in self.transitions:
                trans = self.transitions[source_parts]
                trans.add_output_arc(arc)

        # DEBUG: show place keys and transition connectivity
        logger.debug("[runtime] places=%s", list(self.places.keys()))
        # DEBUG: show transition connectivity
        for tfqn, trans in self.transitions.items():
            try:
                in_count = len(trans.input_arcs)
                out_count = len(trans.output_arcs)
            except Exception:
                in_count = out_count = 0
            logger.debug("[runtime] %s inputs=%d outputs=%d", tfqn, in_count, out_count)

    def _create_interface_view_if_needed(self, bb: Any, handler: Callable) -> Any:
        """
        Create a constrained view if the handler has an interface type hint on its
        blackboard parameter.

        This enables type-safe, constrained access to blackboard state based on
        interface definitions created with @blackboard_interface.

        The function signature can use an interface type:
            async def my_transition(consumed, bb: MyInterface, timebase):
                # bb is automatically a constrained view
                yield {output: token}

        Args:
            bb: The blackboard instance
            handler: The transition or IO handler function to check for interface type hints

        Returns:
            Either the original blackboard or a constrained view based on interface metadata

        Raises:
            TypeError: If handler is not callable or type hints are malformed
            AttributeError: If type hints reference undefined types
        """
        # Get compiled metadata (uses EAFP pattern internally)
        # Raises specific exceptions if compilation fails
        metadata = _get_compiled_metadata(handler)

        # If handler has interface type hint, create constrained view
        if metadata.has_interface and metadata.interface_type:
            # Use instance cache for this runtime
            readonly_fields = metadata.readonly_fields if metadata.readonly_fields else None

            return self._interface_cache.get_or_create(
                bb_id=id(bb),
                interface_type=metadata.interface_type,
                readonly_fields=readonly_fields,
                creator_func=lambda: create_view_from_protocol(
                    bb,
                    metadata.interface_type,
                    readonly_fields=metadata.readonly_fields
                )
            )

        return bb

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring and debugging.

        Returns:
            Dict with current cache size, maxsize, and hit/miss counts
        """
        return self._interface_cache.get_stats()
    
    def _find_place_spec(self, fqn: str) -> PlaceSpec:
        """Find place spec by FQN using hierarchy"""
        parts = fqn.split('.')
        result = self._find_in_spec(self.spec, parts, is_place=True)
        if not isinstance(result, PlaceSpec):
            raise TypeError(f"Expected PlaceSpec, got {type(result)}")
        return result

    def _find_transition_spec(self, fqn: str) -> TransitionSpec:
        """Find transition spec by FQN using hierarchy"""
        parts = fqn.split('.')
        result = self._find_in_spec(self.spec, parts, is_place=False)
        if not isinstance(result, TransitionSpec):
            raise TypeError(f"Expected TransitionSpec, got {type(result)}")
        return result
    
    def _find_in_spec(self, spec: NetSpec, parts: List[str], is_place: bool):
        """Recursively find place or transition in spec hierarchy"""
        # Remove the root name if it matches
        if parts[0] == spec.name:
            parts = parts[1:]
        
        if len(parts) == 1:
            # Leaf - look in places or transitions
            name = parts[0]
            if is_place:
                if name in spec.places:
                    return spec.places[name]
            else:
                if name in spec.transitions:
                    return spec.transitions[name]
            raise KeyError(f"{'Place' if is_place else 'Transition'} {'.'.join([spec.name] + parts)} not found")
        
        # Navigate into subnet
        subnet_name = parts[0]
        if subnet_name in spec.subnets:
            return self._find_in_spec(spec.subnets[subnet_name], parts[1:], is_place)
        
        raise KeyError(f"Subnet {subnet_name} not found in {spec.name}")

    def _to_parts(self, key: Any) -> Tuple[str, ...]:
        """Normalize a dotted string, parts list/tuple, or ref into tuple parts."""
        if isinstance(key, tuple):
            return key
        if isinstance(key, list):
            return tuple(key)
        if hasattr(key, 'get_parts'):
            try:
                return tuple(key.get_parts())
            except Exception:
                pass
        if isinstance(key, str):
            return tuple(key.split('.'))
        # fallback
        return tuple(str(key).split('.'))
    
    def _collect_all_arcs(self) -> List[ArcSpec]:
        """Collect all arcs from spec and subnets recursively"""
        arcs = []
        self._collect_arcs_from_spec(self.spec, arcs)
        return arcs
    
    def _collect_arcs_from_spec(self, spec: NetSpec, arcs: List[ArcSpec]):
        """Recursively collect arcs from spec"""
        arcs.extend(spec.arcs)
        for subnet_spec in spec.subnets.values():
            self._collect_arcs_from_spec(subnet_spec, arcs)

    def _find_place_spec_by_parts(self, parts: Tuple[str, ...]) -> PlaceSpec:
        """Find a PlaceSpec using a tuple of path parts."""
        result = self._find_in_spec(self.spec, list(parts), is_place=True)
        if not isinstance(result, PlaceSpec):
            raise TypeError(f"Expected PlaceSpec, got {type(result)}")
        return result

    def _find_transition_spec_by_parts(self, parts: Tuple[str, ...]) -> TransitionSpec:
        """Find a TransitionSpec using a tuple of path parts."""
        result = self._find_in_spec(self.spec, list(parts), is_place=False)
        if not isinstance(result, TransitionSpec):
            raise TypeError(f"Expected TransitionSpec, got {type(result)}")
        return result
    
    async def start(self):
        """Start the Petri net execution"""
        for place in self.places.values():
            if place.spec.is_io_input:
                await place.start_io_input()
        
        for trans in self.transitions.values():
            trans.task = asyncio.create_task(trans.run())
    
    async def stop(self, timeout: float = 5.0):
        """Stop the Petri net execution"""
        for trans in self.transitions.values():
            await trans.cancel()
        
        for place in self.places.values():
            await place.cancel()
    
    def add_place(self, fqn: str, place_type: PlaceType = PlaceType.BAG, 
                  state_factory: Optional[Callable] = None) -> PlaceRuntime:
        """Dynamically add a place to the running net"""
        parts = self._to_parts(fqn)
        if parts in self.places:
            raise ValueError(f"Place {fqn} already exists")

        place_spec = PlaceSpec(fqn, place_type, state_factory=state_factory)
        place_runtime = PlaceRuntime(place_spec, self.bb, self.timebase, fqn)
        self.places[parts] = place_runtime
        
        self._add_to_spec(place_spec)
        
        return place_runtime
    
    def add_transition(self, fqn: str, handler: Callable, 
                      guard: Optional[GuardSpec] = None,
                      state_factory: Optional[Callable] = None) -> TransitionRuntime:
        """Dynamically add a transition to the running net"""
        parts = self._to_parts(fqn)
        if parts in self.transitions:
            raise ValueError(f"Transition {fqn} already exists")

        trans_spec = TransitionSpec(fqn, handler, guard, state_factory)
        trans_runtime = TransitionRuntime(trans_spec, self, fqn)
        self.transitions[parts] = trans_runtime
        
        trans_runtime.task = asyncio.create_task(trans_runtime.run())
        
        self._add_to_spec(trans_spec)
        
        return trans_runtime
    
    def add_arc(self, source_fqn: str, target_fqn: str, weight: int = 1, 
                name: Optional[str] = None):
        """Dynamically add an arc to the running net"""
        if source_fqn not in self.places and source_fqn not in self.transitions:
            raise ValueError(f"Source {source_fqn} does not exist")
        if target_fqn not in self.places and target_fqn not in self.transitions:
            raise ValueError(f"Target {target_fqn} does not exist")
        
        arc_name = name or f"{source_fqn}->{target_fqn}"
        arc_spec = ArcSpec(source_fqn, target_fqn, weight, arc_name)
        
        if target_fqn in self.transitions:
            trans = self.transitions[target_fqn]
            trans.add_input_arc(source_fqn, arc_spec)
        
        if source_fqn in self.transitions:
            trans = self.transitions[source_fqn]
            trans.add_output_arc(arc_spec)
        
        self._add_arc_to_spec(arc_spec)
    
    async def remove_arc(self, source_fqn: str, target_fqn: str):
        """Dynamically remove an arc from the running net"""
        arc_to_remove = None
        
        for arc in self._collect_all_arcs():
            if arc.source == source_fqn and arc.target == target_fqn:
                arc_to_remove = arc
                break
        
        if not arc_to_remove:
            raise ValueError(f"Arc from {source_fqn} to {target_fqn} does not exist")
        
        if target_fqn in self.transitions:
            trans = self.transitions[target_fqn]
            
            await trans.cancel()
            
            trans.input_arcs = [(p, a) for p, a in trans.input_arcs 
                               if not (a.source == source_fqn and a.target == target_fqn)]
            
            if trans.input_arcs:
                trans._stop_event.clear()
                trans.task = asyncio.create_task(trans.run())
        
        if source_fqn in self.transitions:
            trans = self.transitions[source_fqn]
            trans.output_arcs = [a for a in trans.output_arcs 
                                if not (a.source == source_fqn and a.target == target_fqn)]
        
        self._remove_arc_from_spec(arc_to_remove)
    
    async def remove(self, fqn: str):
        """Remove a place or transition by fully qualified name"""
        if fqn in self.places:
            place = self.places[fqn]
            await place.cancel()
            
            arcs_to_remove = [arc for arc in self._collect_all_arcs() 
                            if arc.source == fqn or arc.target == fqn]
            
            affected_transitions = set()
            
            for arc in arcs_to_remove:
                if arc.target in self.transitions:
                    trans = self.transitions[arc.target]
                    affected_transitions.add(trans)
                    trans.input_arcs = [(p, a) for p, a in trans.input_arcs if a != arc]
                
                if arc.source in self.transitions:
                    trans = self.transitions[arc.source]
                    trans.output_arcs = [a for a in trans.output_arcs if a != arc]
                
                self._remove_arc_from_spec(arc)
            
            for trans in affected_transitions:
                await trans.cancel()
                if trans.input_arcs:
                    trans._stop_event.clear()
                    trans.task = asyncio.create_task(trans.run())
            
            del self.places[fqn]
            self._remove_from_spec(fqn, is_place=True)
        
        elif fqn in self.transitions:
            trans = self.transitions[fqn]
            await trans.cancel()
            
            arcs_to_remove = [arc for arc in self._collect_all_arcs() 
                            if arc.source == fqn or arc.target == fqn]
            
            for arc in arcs_to_remove:
                self._remove_arc_from_spec(arc)
            
            del self.transitions[fqn]
            self._remove_from_spec(fqn, is_place=False)
        
        else:
            raise ValueError(f"{fqn} not found in places or transitions")
    
    def _add_to_spec(self, spec):
        """Add place or transition spec to appropriate location"""
        if isinstance(spec, PlaceSpec):
            self._add_to_nested_spec(spec.name, spec, is_place=True)
        elif isinstance(spec, TransitionSpec):
            self._add_to_nested_spec(spec.name, spec, is_place=False)
    
    def _add_to_nested_spec(self, fqn: str, spec, is_place: bool):
        """Add spec to correct nested location based on FQN"""
        parts = fqn.split('.')
        current_spec = self.spec
        
        for i, part in enumerate(parts[1:-1], start=1):
            subnet_fqn = '.'.join(parts[:i+1])
            if subnet_fqn not in current_spec.subnets:
                return
            current_spec = current_spec.subnets[subnet_fqn]
        
        if is_place:
            current_spec.places[fqn] = spec
        else:
            current_spec.transitions[fqn] = spec
    
    def _add_arc_to_spec(self, arc_spec: ArcSpec):
        """Add arc to appropriate spec level"""
        # arc_spec.source may be a ref or string; use precomputed parts
        parts = list(arc_spec.source_parts)
        current_spec = self.spec
        
        for i, part in enumerate(parts[1:-1], start=1):
            subnet_fqn = '.'.join(parts[:i+1])
            if subnet_fqn in current_spec.subnets:
                current_spec = current_spec.subnets[subnet_fqn]
            else:
                break
        
        current_spec.arcs.append(arc_spec)
    
    def _remove_from_spec(self, fqn: str, is_place: bool):
        """Remove place or transition from spec"""
        parts = fqn.split('.')
        current_spec = self.spec
        
        for i, part in enumerate(parts[1:-1], start=1):
            subnet_fqn = '.'.join(parts[:i+1])
            if subnet_fqn not in current_spec.subnets:
                return
            current_spec = current_spec.subnets[subnet_fqn]
        
        if is_place and fqn in current_spec.places:
            del current_spec.places[fqn]
        elif not is_place and fqn in current_spec.transitions:
            del current_spec.transitions[fqn]
    
    def _remove_arc_from_spec(self, arc: ArcSpec):
        """Remove arc from spec"""
        def remove_from_spec_recursive(spec: NetSpec):
            spec.arcs = [a for a in spec.arcs 
                        if not (a.source == arc.source and a.target == arc.target)]
            for subnet_spec in spec.subnets.values():
                remove_from_spec_recursive(subnet_spec)
        
        remove_from_spec_recursive(self.spec)


class Runner:
    """High-level runner for executing a Petri net"""
    
    def __init__(self, net_func: Any, blackboard: Any):
        if not hasattr(net_func, '_spec'):
            raise ValueError(f"{net_func} is not a valid net")
        
        self.spec = net_func._spec
        self.blackboard = blackboard
        self.timebase = None
        self.runtime: Optional[NetRuntime] = None
    
    async def start(self, timebase: Any):
        """Start the net with given timebase"""
        self.timebase = timebase
        self.runtime = NetRuntime(self.spec, self.blackboard, self.timebase)
        await self.runtime.start()
    
    async def stop(self, timeout: float = 5.0):
        """Stop the net"""
        if self.runtime:
            await self.runtime.stop(timeout)
    
    def add_place(self, fqn: str, place_type: PlaceType = PlaceType.BAG, 
                  state_factory: Optional[Callable] = None):
        """Add a place to the running net"""
        if not self.runtime:
            raise RuntimeError("Net is not running. Call start() first.")
        return self.runtime.add_place(fqn, place_type, state_factory)
    
    def add_transition(self, fqn: str, handler: Callable, 
                      guard: Optional[GuardSpec] = None,
                      state_factory: Optional[Callable] = None):
        """Add a transition to the running net"""
        if not self.runtime:
            raise RuntimeError("Net is not running. Call start() first.")
        return self.runtime.add_transition(fqn, handler, guard, state_factory)
    
    def add_arc(self, source_fqn: str, target_fqn: str, weight: int = 1, 
                name: Optional[str] = None):
        """Add an arc to the running net"""
        if not self.runtime:
            raise RuntimeError("Net is not running. Call start() first.")
        self.runtime.add_arc(source_fqn, target_fqn, weight, name)
    
    async def remove_arc(self, source_fqn: str, target_fqn: str):
        """Remove an arc from the running net"""
        if not self.runtime:
            raise RuntimeError("Net is not running. Call start() first.")
        await self.runtime.remove_arc(source_fqn, target_fqn)
    
    async def remove(self, fqn: str):
        """Remove a place or transition"""
        if not self.runtime:
            raise RuntimeError("Net is not running. Call start() first.")
        await self.runtime.remove(fqn)