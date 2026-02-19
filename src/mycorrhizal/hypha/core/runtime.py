#!/usr/bin/env python3
"""
Hypha DSL - Runtime Layer

Runtime objects that execute the Petri net specification.
Manages token flow, transition firing, and asyncio task coordination.
"""

import asyncio
from asyncio import Event, Task
from typing import Any, List, Dict, Optional, Tuple, Callable
from itertools import product, combinations
import inspect
import logging
import time
from dataclasses import dataclass, field

from mycorrhizal.common.wrappers import create_view_from_protocol
from mycorrhizal.common.compilation import (
    _get_compiled_metadata,
)
from mycorrhizal.common.cache import InterfaceViewCache
from .specs import NetSpec, PlaceSpec, TransitionSpec, ArcSpec, GuardSpec

logger = logging.getLogger(__name__)


try:
    # Library should not configure root logging; be quiet by default
    logger.addHandler(logging.NullHandler())
except Exception:
    # NullHandler may not be available in very old Pythons; ignore if so
    pass
# Library does not configure handlers by default. Callers may configure logging.


# =============================================================================
# Matrix-Based Runtime Layer
# =============================================================================

"""
Matrix multiplication-style execution for Petri nets. Replaces per-transition
asyncio coordination with batch operations using incidence matrices.

Core execution: M_new = M + (A @ F)
- M: State vector (marking)
- A: Incidence matrix (net structure)
- F: Firing vector (which transitions fire)

This eliminates the ~220x overhead from per-transition asyncio task coordination.

Synchronous Execution Mode:
- For pure computational nets without async features, use run_sync() for minimal overhead
- Async features (IO places, async transitions/guards) require async mode
"""


class PlaceWrapper:
    """Wrapper for places in MatrixRuntime to provide NetRuntime-like API."""

    def __init__(self, matrix_runtime: 'MatrixRuntime', place_idx: int):
        self._runtime = matrix_runtime
        self._place_idx = place_idx
        # Compatibility: provide token_added_event like NetRuntime
        self.token_added_event = asyncio.Event()

    @property
    def tokens(self) -> List[Any]:
        """Get tokens at this place."""
        token_ids = self._runtime.marking.tokens.get(self._place_idx, [])
        return [self._runtime.token_registry.get(tid) for tid in token_ids]

    def add_token(self, token: Any):
        """Add a token to this place."""
        self._runtime.add_token(self._place_idx, token)
        # Set the event to notify that a token was added
        self.token_added_event.set()


# =============================================================================
# Incidence Matrix Representation
# =============================================================================

@dataclass
class IncidenceMatrix:
    """Incidence matrix representation of Petri net structure.

    The incidence matrix A represents the net structure:
    - Rows: places (P rows)
    - Columns: transitions (T columns)
    - Entry A[i,j]: net token change at place i when transition j fires
      - Negative value: tokens consumed from place i
      - Positive value: tokens produced to place i
      - Zero: no connection

    For a transition with input arcs (consuming tokens) and output arcs
    (producing tokens), the column shows the net change.
    """
    # Number of places and transitions
    num_places: int
    num_transitions: int

    # Matrix stored as dict of dict for sparse representation
    # matrix[place_idx][transition_idx] = net_token_change
    matrix: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # Mapping from FQN tuples to indices
    place_to_idx: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    idx_to_place: Dict[int, Tuple[str, ...]] = field(default_factory=dict)

    transition_to_idx: Dict[Tuple[str, ...], int] = field(default_factory=dict)
    idx_to_transition: Dict[int, Tuple[str, ...]] = field(default_factory=dict)

    # Track input requirements for each transition (for enabling check)
    # input_requirements[trans_idx] = {place_idx: required_tokens}
    input_requirements: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # Track outputs for each transition (for token data routing)
    # output_destinations[trans_idx] = {place_idx: produced_tokens}
    output_destinations: Dict[int, Dict[int, int]] = field(default_factory=dict)

    # Cache token slots for each transition (computed once, reused)
    # _token_slots_cache[trans_idx] = list of (place_idx, slot_idx) tuples
    _token_slots_cache: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict, init=False)

    def get(self, place_idx: int, trans_idx: int) -> int:
        """Get matrix entry at (place_idx, trans_idx)"""
        return self.matrix.get(place_idx, {}).get(trans_idx, 0)

    def set(self, place_idx: int, trans_idx: int, value: int):
        """Set matrix entry at (place_idx, trans_idx)"""
        if place_idx not in self.matrix:
            self.matrix[place_idx] = {}
        self.matrix[place_idx][trans_idx] = value

    def add_input(self, place_idx: int, trans_idx: int, weight: int):
        """Add input arc (consumes tokens, so negative entry)"""
        current = self.get(place_idx, trans_idx)
        self.set(place_idx, trans_idx, current - weight)

        # Track input requirements (sum weights for multiple arcs)
        if trans_idx not in self.input_requirements:
            self.input_requirements[trans_idx] = {}
        if place_idx not in self.input_requirements[trans_idx]:
            self.input_requirements[trans_idx][place_idx] = 0
        self.input_requirements[trans_idx][place_idx] += weight

    def add_output(self, place_idx: int, trans_idx: int, weight: int):
        """Add output arc (produces tokens, so positive entry)"""
        current = self.get(place_idx, trans_idx)
        self.set(place_idx, trans_idx, current + weight)

        # Track output destinations (sum weights for multiple arcs)
        if trans_idx not in self.output_destinations:
            self.output_destinations[trans_idx] = {}
        if place_idx not in self.output_destinations[trans_idx]:
            self.output_destinations[trans_idx][place_idx] = 0
        self.output_destinations[trans_idx][place_idx] += weight

    def compute_state_change(self, firing_vector: List[int]) -> Dict[int, int]:
        """Compute state change: delta = A @ F

        Args:
            firing_vector: List of length num_transitions with 0/1 values

        Returns:
            Dict mapping place_idx -> net_token_change
        """
        state_change: Dict[int, int] = {}

        for place_idx, trans_entries in self.matrix.items():
            delta = 0
            for trans_idx, value in trans_entries.items():
                delta += value * firing_vector[trans_idx]

            if delta != 0:
                state_change[place_idx] = delta

        return state_change

    def is_enabled(self, trans_idx: int, marking: Dict[int, int]) -> bool:
        """Check if transition is enabled (has sufficient input tokens).

        IMPORTANT: When multiple arcs connect the same place to a transition,
        we need to check if there are enough tokens to satisfy all arcs.
        The old runtime allowed "bag semantics" where arcs could consume the
        same token, but for practical purposes we require distinct tokens per arc.

        Args:
            trans_idx: Transition index
            marking: Current marking (place_idx -> token_count)

        Returns:
            True if transition has all required input tokens
        """
        if trans_idx not in self.input_requirements:
            # No input requirements (generator transition) - always enabled
            return True

        for place_idx, required in self.input_requirements[trans_idx].items():
            # Check if place has enough tokens to satisfy all arcs
            # For bag semantics, we allow the same token to be consumed by multiple arcs,
            # so we only need 1 token minimum. However, to avoid infinite loops when
            # transitions consume more than they produce, we check for required tokens.
            if marking.get(place_idx, 0) < required:
                return False

        return True

    def get_token_slots(self, trans_idx: int) -> List[Tuple[int, int]]:
        """Get token consumption slots for a transition (cached for performance).

        When multiple arcs connect the same place to a transition,
        we need to consume separate tokens (or the same token multiple times for bag semantics).

        Returns:
            List of (place_idx, slot_index) tuples representing token consumption slots
        """
        # Return cached slots if available
        if trans_idx in self._token_slots_cache:
            return self._token_slots_cache[trans_idx]

        if trans_idx not in self.input_requirements:
            return []

        slots = []
        for place_idx, count in self.input_requirements[trans_idx].items():
            for i in range(count):
                slots.append((place_idx, i))

        # Cache for future use
        self._token_slots_cache[trans_idx] = slots
        return slots


def build_incidence_matrix(spec: NetSpec) -> IncidenceMatrix:
    """Build incidence matrix from net specification.

    Args:
        spec: Net specification

    Returns:
        IncidenceMatrix representation
    """
    # Collect all places, transitions, and arcs
    places: Dict[Tuple[str, ...], PlaceSpec] = {}
    transitions: Dict[Tuple[str, ...], TransitionSpec] = {}
    arcs: List[ArcSpec] = []

    def collect_spec(net_spec: NetSpec):
        for place_name, place_spec in net_spec.places.items():
            places[tuple(net_spec.get_parts(place_name))] = place_spec

        for trans_name, trans_spec in net_spec.transitions.items():
            transitions[tuple(net_spec.get_parts(trans_name))] = trans_spec

        arcs.extend(net_spec.arcs)

        for subnet_spec in net_spec.subnets.values():
            collect_spec(subnet_spec)

    collect_spec(spec)

    # Create index mappings
    place_to_idx = {fqn: idx for idx, fqn in enumerate(sorted(places.keys()))}
    idx_to_place = {idx: fqn for fqn, idx in place_to_idx.items()}

    transition_to_idx = {fqn: idx for idx, fqn in enumerate(sorted(transitions.keys()))}
    idx_to_transition = {idx: fqn for fqn, idx in transition_to_idx.items()}

    # Build incidence matrix
    matrix = IncidenceMatrix(
        num_places=len(places),
        num_transitions=len(transitions),
        place_to_idx=place_to_idx,
        idx_to_place=idx_to_place,
        transition_to_idx=transition_to_idx,
        idx_to_transition=idx_to_transition,
    )

    # Process arcs
    for arc in arcs:
        source_parts = tuple(arc.source_parts)
        target_parts = tuple(arc.target_parts)

        # Place -> Transition (input arc)
        if source_parts in place_to_idx and target_parts in transition_to_idx:
            place_idx = place_to_idx[source_parts]
            trans_idx = transition_to_idx[target_parts]
            matrix.add_input(place_idx, trans_idx, arc.weight)

        # Transition -> Place (output arc)
        elif source_parts in transition_to_idx and target_parts in place_to_idx:
            trans_idx = transition_to_idx[source_parts]
            place_idx = place_to_idx[target_parts]
            matrix.add_output(place_idx, trans_idx, arc.weight)

    return matrix


# =============================================================================
# State Vector and Token Management
# =============================================================================

@dataclass
class TokenEntry:
    """Represents a token with optional data payload."""
    token_id: int
    data: Any = None


class TokenRegistry:
    """Registry for tracking token data.

    Since incidence matrices track counts, we need a separate registry
    to map token identifiers to their actual data payloads.
    """
    def __init__(self):
        self._next_id: int = 0
        self._tokens: Dict[int, Any] = {}

    def register(self, data: Any) -> int:
        """Register a token and return its ID."""
        token_id = self._next_id
        self._next_id += 1
        self._tokens[token_id] = data
        return token_id

    def get(self, token_id: int) -> Any:
        """Get token data by ID."""
        return self._tokens.get(token_id)

    def remove(self, token_id: int):
        """Remove a token from registry."""
        if token_id in self._tokens:
            del self._tokens[token_id]


@dataclass
class Marking:
    """State vector representation of Petri net marking.

    For data-carrying tokens, stores token IDs. For simple count places,
    stores the count directly.

    PERFORMANCE: Cached count dict is invalidated on token changes to avoid
    recomputing on every transition check.
    """
    # place_idx -> list of token IDs (or count for simple places)
    tokens: Dict[int, List[Any]] = field(default_factory=dict)
    _cached_count_dict: Optional[Dict[int, int]] = field(default=None, init=False)
    _cache_valid: bool = field(default=False, init=False)

    def _invalidate_cache(self):
        """Invalidate the count cache."""
        self._cache_valid = False

    def get_count(self, place_idx: int) -> int:
        """Get token count at a place."""
        return len(self.tokens.get(place_idx, []))

    def add_tokens(self, place_idx: int, tokens: List[Any]):
        """Add tokens to a place."""
        if place_idx not in self.tokens:
            self.tokens[place_idx] = []
        self.tokens[place_idx].extend(tokens)
        self._invalidate_cache()

    def remove_tokens(self, place_idx: int, count: int) -> List[Any]:
        """Remove tokens from a place (returns removed tokens)."""
        if place_idx not in self.tokens:
            return []

        place_tokens = self.tokens[place_idx]
        removed = place_tokens[:count]
        self.tokens[place_idx] = place_tokens[count:]

        if not self.tokens[place_idx]:
            del self.tokens[place_idx]

        self._invalidate_cache()
        return removed

    def remove_tokens_fast(self, place_idx: int, token_ids: List[Any]):
        """Remove specific tokens from a place (faster O(1) per token).

        This is used when we know exactly which tokens to remove.
        Uses a set-based approach for O(n) total instead of O(n*m).
        """
        if place_idx not in self.tokens:
            return

        if not token_ids:
            return

        place_tokens = self.tokens[place_idx]

        # Build a set of tokens to remove for O(1) lookup
        to_remove = set(token_ids)

        # Filter out the removed tokens (single pass)
        self.tokens[place_idx] = [t for t in place_tokens if t not in to_remove]

        if not self.tokens[place_idx]:
            del self.tokens[place_idx]

        self._invalidate_cache()

    def peek_tokens(self, place_idx: int, count: int) -> List[Any]:
        """Peek at tokens without removing."""
        place_tokens = self.tokens.get(place_idx, [])
        return place_tokens[:count]

    def has_tokens(self, place_idx: int, count: int) -> bool:
        """Check if place has at least count tokens."""
        return self.get_count(place_idx) >= count

    def get_count_dict(self) -> Dict[int, int]:
        """Get dict of place_idx -> token_count (cached for performance)."""
        if not self._cache_valid:
            self._cached_count_dict = {idx: len(tokens) for idx, tokens in self.tokens.items()}
            self._cache_valid = True
        return self._cached_count_dict


# =============================================================================
# Guard Type Matrix
# =============================================================================

@dataclass
class GuardInfo:
    """Information about a transition's guard."""
    has_guard: bool = False
    guard_spec: Optional[GuardSpec] = None
    requires_tokens: bool = False  # True if guard needs token data


class GuardMatrix:
    """Guard type matrix and evaluation logic.

    Tracks which transitions have guards and evaluates them to build
    the firing vector F.
    """
    def __init__(self):
        # trans_idx -> GuardInfo
        self.guards: Dict[int, GuardInfo] = {}

    def set_guard(self, trans_idx: int, guard_spec: Optional[GuardSpec], requires_tokens: bool = False):
        """Set guard info for a transition."""
        self.guards[trans_idx] = GuardInfo(
            has_guard=guard_spec is not None,
            guard_spec=guard_spec,
            requires_tokens=requires_tokens,
        )

    def has_guard(self, trans_idx: int) -> bool:
        """Check if transition has a guard."""
        return self.guards.get(trans_idx, GuardInfo()).has_guard

    def evaluate(
        self,
        trans_idx: int,
        marking: Marking,
        incidence_matrix: IncidenceMatrix,
        bb: Any,
        timebase: Any,
        token_registry: TokenRegistry,
    ) -> bool | Any:
        """Evaluate guard for a transition.

        Args:
            trans_idx: Transition index
            marking: Current marking
            incidence_matrix: Incidence matrix for net structure
            bb: Blackboard
            timebase: Timebase
            token_registry: Token registry for data lookups

        Returns:
            True if guard passes (or no guard), False otherwise
            Can also return async generator for async guards (handled by caller)
        """
        guard_info = self.guards.get(trans_idx)
        if not guard_info or not guard_info.has_guard:
            return True

        guard_spec = guard_info.guard_spec
        if guard_spec is None:
            return True

        # Get input places for this transition
        input_places = incidence_matrix.input_requirements.get(trans_idx, {})

        if not input_places:
            # Generator transition - no tokens to evaluate
            combinations = [[]]
        else:
            # Generate token combinations
            arc_tokens = []
            for place_idx, required_count in input_places.items():
                tokens = marking.peek_tokens(place_idx, required_count)
                if len(tokens) < required_count:
                    return False  # Not enough tokens

                if required_count == 1:
                    arc_tokens.append([(t,) for t in tokens])
                else:
                    arc_tokens.append(list(combinations(tokens, required_count)))

            if not arc_tokens:
                return False

            combinations = list(product(*arc_tokens))

        if not combinations:
            return False

        # Evaluate guard
        guard_func = guard_spec.func
        guard_result = guard_func(combinations, bb, timebase)

        # Check if guard passed any combination
        if inspect.isgenerator(guard_result):
            for result in guard_result:
                if result is not None:
                    return True
            return False
        elif inspect.isasyncgen(guard_result):
            # Need to handle async guard - return coroutine for caller to await
            return guard_result
        else:
            return guard_result is not None


def build_guard_matrix(spec: NetSpec, incidence_matrix: IncidenceMatrix) -> GuardMatrix:
    """Build guard matrix from net specification.

    Args:
        spec: Net specification
        incidence_matrix: Incidence matrix for index mapping

    Returns:
        GuardMatrix with guard information
    """
    guard_matrix = GuardMatrix()

    transitions: Dict[Tuple[str, ...], TransitionSpec] = {}

    def collect_spec(net_spec: NetSpec):
        for trans_name, trans_spec in net_spec.transitions.items():
            transitions[tuple(net_spec.get_parts(trans_name))] = trans_spec
        for subnet_spec in net_spec.subnets.values():
            collect_spec(subnet_spec)

    collect_spec(spec)

    for trans_fqn, trans_spec in transitions.items():
        trans_idx = incidence_matrix.transition_to_idx.get(trans_fqn)
        if trans_idx is not None:
            guard_matrix.set_guard(trans_idx, trans_spec.guard)

    return guard_matrix


# =============================================================================
# Matrix Runtime
# =============================================================================

class MatrixRuntime:
    """Matrix-based Petri net runtime.

    Uses incidence matrix and state vector for execution, eliminating
    per-transition asyncio overhead.

    Execution model:
    1. Build firing vector F by evaluating enabled transitions
    2. Fire via M_new = M + (A @ F)
    3. Process token data flow for fired transitions
    4. Repeat
    """

    def __init__(self, spec: NetSpec, bb: Any, timebase: Any):
        self.spec = spec
        self.bb = bb
        self.timebase = timebase
        self._interface_cache = InterfaceViewCache(maxsize=256)

        # Compatibility layer: places dict accessor
        self._places_wrapper: Dict[Tuple[str, ...], 'PlaceWrapper'] = {}

        # Build matrix representations
        self.incidence_matrix = build_incidence_matrix(spec)
        self.guard_matrix = build_guard_matrix(spec, self.incidence_matrix)

        # State
        self.marking = Marking()
        self.token_registry = TokenRegistry()
        self._stop_event = asyncio.Event()
        self._stop_event_sync = False  # Synchronous stop flag

        # Transition specs for execution
        self._transition_specs: Dict[int, TransitionSpec] = {}
        self._transition_states: Dict[int, Any] = {}

        # Delay tracking
        self._enabled_times: Dict[int, float] = {}

        # Place specs for IO handling (initialize early for async detection)
        self._place_specs: Dict[int, PlaceSpec] = {}

        # Collect transition specs and place specs first (needed for async detection)
        transitions: Dict[Tuple[str, ...], TransitionSpec] = {}
        places: Dict[Tuple[str, ...], PlaceSpec] = {}

        def collect_spec(net_spec: NetSpec):
            for trans_name, trans_spec in net_spec.transitions.items():
                transitions[tuple(net_spec.get_parts(trans_name))] = trans_spec
            for place_name, place_spec in net_spec.places.items():
                places[tuple(net_spec.get_parts(place_name))] = place_spec
            for subnet_spec in net_spec.subnets.values():
                collect_spec(subnet_spec)

        collect_spec(spec)

        # Populate transition specs
        for trans_fqn, trans_spec in transitions.items():
            trans_idx = self.incidence_matrix.transition_to_idx.get(trans_fqn)
            if trans_idx is not None:
                self._transition_specs[trans_idx] = trans_spec
                if trans_spec.state_factory:
                    self._transition_states[trans_idx] = trans_spec.state_factory()

        # Populate place specs (done before async detection)
        for place_fqn, place_spec in places.items():
            place_idx = self.incidence_matrix.place_to_idx.get(place_fqn)
            if place_idx is not None:
                self._place_specs[place_idx] = place_spec

        # Detect if net has async features (IO places, async transitions/guards)
        self._has_async_features = self._detect_async_features()

        # Build places wrapper for compatibility
        for place_fqn in self.incidence_matrix.place_to_idx.keys():
            place_idx = self.incidence_matrix.place_to_idx[place_fqn]
            self._places_wrapper[place_fqn] = PlaceWrapper(self, place_idx)

        # IO tasks
        self._io_tasks: List[asyncio.Task] = []
        self._run_task: Optional[asyncio.Task] = None

    def _detect_async_features(self) -> bool:
        """Detect if the net has async features that require async mode.

        Returns:
            True if net has IO places, async transitions, or async guards
        """
        # Check for IO places
        for place_spec in self._place_specs.values():
            if place_spec.is_io_input or place_spec.is_io_output:
                return True

        # Check for async transitions or guards
        for trans_spec in self._transition_specs.values():
            # Check if transition handler is async
            if inspect.iscoroutinefunction(trans_spec.handler):
                return True

            # Check if guard is async
            if trans_spec.guard and inspect.isasyncgenfunction(trans_spec.guard.func):
                return True

        return False

    @property
    def places(self) -> Dict[Tuple[str, ...], PlaceWrapper]:
        """Get places dict (compatibility layer with NetRuntime)."""
        return self._places_wrapper

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring and debugging.

        Returns:
            Dict with current cache size, maxsize, and hit/miss counts
        """
        return self._interface_cache.get_stats()

    @property
    def transitions(self) -> Dict[str, Any]:
        """Get transitions dict (compatibility layer with NetRuntime).

        Note: This returns a simplified view for compatibility with mycelium bridge.
        The actual transition data is stored in _transition_specs.
        """
        # Create the wrapper dict on first access and cache it
        if not hasattr(self, '_transitions_wrapper'):
            self._transitions_wrapper = {}
            for trans_idx, trans_spec in self._transition_specs.items():
                trans_fqn = self.incidence_matrix.idx_to_transition.get(trans_idx)
                if trans_fqn:
                    # Create a wrapper object that provides the attributes expected by the bridge
                    # The bridge needs: spec, input_arcs, output_arcs
                    input_arcs = self.incidence_matrix.input_requirements.get(trans_idx, {})
                    output_arcs = self.incidence_matrix.output_destinations.get(trans_idx, {})

                    # Convert the arc dictionaries to a format that matches NetRuntime's structure
                    # NetRuntime stores arcs as ArcSpec objects, we need to provide something compatible
                    class TransitionWrapper:
                        def __init__(self, spec, input_arcs_dict, output_arcs_dict, incidence_matrix, trans_idx):
                            self.spec = spec
                            self._input_arcs_dict = input_arcs_dict
                            self._output_arcs_dict = output_arcs_dict
                            self._incidence_matrix = incidence_matrix
                            self._trans_idx = trans_idx
                            # Create fake input_arcs list for compatibility
                            # Format: list of (place_parts, arc_spec) tuples
                            self.input_arcs = []
                            self.output_arcs = []

                            # Convert input requirements to fake ArcSpec objects
                            for place_idx, count in input_arcs_dict.items():
                                place_fqn = incidence_matrix.idx_to_place.get(trans_idx, ('unknown',))
                                # Create a minimal ArcSpec-like object
                                class FakeArcSpec:
                                    def __init__(self, source, target, weight):
                                        self.source = source
                                        self.target = target
                                        self.weight = weight
                                arc = FakeArcSpec(place_idx, place_fqn, count)
                                self.input_arcs.append((incidence_matrix.idx_to_place.get(place_idx, ('unknown',)), arc))

                            # Convert output destinations to fake ArcSpec objects
                            for place_idx, count in output_arcs_dict.items():
                                place_fqn = incidence_matrix.idx_to_transition.get(trans_idx, ('unknown',))
                                class FakeArcSpec:
                                    def __init__(self, source, target, weight):
                                        self.source = source
                                        self.target = target
                                        self.weight = weight
                                arc = FakeArcSpec(place_fqn, place_idx, count)
                                self.output_arcs.append(arc)

                            # Task attribute for compatibility (None since matrix runtime doesn't use per-transition tasks)
                            self.task = None

                        def get_spurious_wakeup_count(self):
                            """Return 0 since matrix runtime doesn't have spurious wakeups."""
                            return 0

                    self._transitions_wrapper[trans_fqn] = TransitionWrapper(
                        trans_spec, input_arcs, output_arcs, self.incidence_matrix, trans_idx
                    )

        return self._transitions_wrapper

    def _create_interface_view_if_needed(self, bb: Any, handler: Callable) -> Any:
        """Create constrained view if handler has interface type hint."""
        metadata = _get_compiled_metadata(handler)

        if metadata.has_interface and metadata.interface_type:
            readonly_fields = tuple(metadata.readonly_fields) if metadata.readonly_fields else None

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

    async def _start_io_input(self):
        """Start IO input places."""
        for place_idx, place_spec in self._place_specs.items():
            if place_spec.is_io_input and place_spec.handler:
                task = asyncio.create_task(self._io_input_loop(place_idx, place_spec))
                self._io_tasks.append(task)

    async def _io_input_loop(self, place_idx: int, place_spec: PlaceSpec):
        """IO input loop for a single place."""
        try:
            handler = place_spec.handler
            if handler is None:
                return

            bb_to_pass = self._create_interface_view_if_needed(self.bb, handler)

            sig = inspect.signature(handler)
            if len(sig.parameters) == 2:
                gen = handler(bb_to_pass, self.timebase)
            else:
                gen = handler()  # type: ignore[call-arg]

            async for token in gen:
                token_id = self.token_registry.register(token)
                self.marking.add_tokens(place_idx, [token_id])
        except asyncio.CancelledError:
            pass

    async def _handle_io_output(self, place_idx: int, token: Any):
        """Handle IO output for a token."""
        place_spec = self._place_specs.get(place_idx)
        if place_spec and place_spec.is_io_output and place_spec.handler:
            bb_to_pass = self._create_interface_view_if_needed(self.bb, place_spec.handler)

            sig = inspect.signature(place_spec.handler)
            if len(sig.parameters) == 3:
                await place_spec.handler(token, bb_to_pass, self.timebase)
            else:
                await place_spec.handler(token)

    def add_token(self, place_idx: int, token: Any):
        """Add a token to a place."""
        if isinstance(token, int) and token in self.token_registry._tokens:
            # Already a token ID
            token_id = token
        else:
            token_id = self.token_registry.register(token)
        self.marking.add_tokens(place_idx, [token_id])

    def add_token_by_fqn(self, place_fqn: Tuple[str, ...], token: Any):
        """Add a token by place FQN."""
        place_idx = self.incidence_matrix.place_to_idx.get(place_fqn)
        if place_idx is not None:
            self.add_token(place_idx, token)

    def get_token_count_by_fqn(self, place_fqn: Tuple[str, ...]) -> int:
        """Get token count by place FQN."""
        place_idx = self.incidence_matrix.place_to_idx.get(place_fqn)
        if place_idx is not None:
            return self.marking.get_count(place_idx)
        return 0

    async def _build_firing_vector(self) -> List[int]:
        """Build firing vector F by evaluating enabled transitions.

        Returns:
            List of 0/1 values indicating which transitions fire
        """
        firing_vector = [0] * self.incidence_matrix.num_transitions

        # Get count dict once per cycle (cached in Marking)
        count_dict = self.marking.get_count_dict()

        for trans_idx in range(self.incidence_matrix.num_transitions):
            # Check if enabled (has sufficient tokens)
            if not self.incidence_matrix.is_enabled(trans_idx, count_dict):
                self._enabled_times.pop(trans_idx, None)
                continue

            # Track enabled time for delay support
            trans_spec = self._transition_specs.get(trans_idx)
            if trans_spec and trans_spec.delay > 0:
                if trans_idx not in self._enabled_times:
                    self._enabled_times[trans_idx] = self.timebase.now()

                # Check if delay has elapsed
                enabled_time = self._enabled_times.get(trans_idx)
                if enabled_time is None:
                    continue

                elapsed = self.timebase.now() - enabled_time
                if elapsed < trans_spec.delay:
                    continue

            # Evaluate guard
            guard_result = self.guard_matrix.evaluate(
                trans_idx,
                self.marking,
                self.incidence_matrix,
                self.bb,
                self.timebase,
                self.token_registry,
            )

            # Handle async guard
            if inspect.isasyncgen(guard_result):
                guard_passed = False
                async for result in guard_result:
                    if result is not None:
                        guard_passed = True
                        break
                if not guard_passed:
                    continue
            elif isinstance(guard_result, bool) and not guard_result:
                continue
            elif not guard_result:
                continue

            # Transition fires!
            firing_vector[trans_idx] = 1

        return firing_vector

    async def _execute_transition(self, trans_idx: int, input_tokens: List[Any]):
        """Execute a transition with input tokens.

        Args:
            trans_idx: Transition index
            input_tokens: List of consumed token data
        """
        trans_spec = self._transition_specs.get(trans_idx)
        if not trans_spec:
            return

        # Create interface view if needed
        bb_to_pass = self._create_interface_view_if_needed(self.bb, trans_spec.handler)

        # Execute transition
        trans_state = self._transition_states.get(trans_idx)

        if trans_state is not None:
            logger.debug("[matrix_fire] trans=%s consumed=%s",
                        self.incidence_matrix.idx_to_transition[trans_idx], input_tokens)
            results = trans_spec.handler(input_tokens, bb_to_pass, self.timebase, trans_state)
        else:
            logger.debug("[matrix_fire] trans=%s consumed=%s",
                        self.incidence_matrix.idx_to_transition[trans_idx], input_tokens)
            results = trans_spec.handler(input_tokens, bb_to_pass, self.timebase)

        # Process results
        if inspect.isasyncgen(results):
            async for yielded in results:
                await self._process_yield(trans_idx, yielded)
        elif inspect.iscoroutine(results):
            result = await results
            if result is not None:
                await self._process_yield(trans_idx, result)
        else:
            await self._process_yield(trans_idx, results)

    async def _process_yield(self, trans_idx: int, yielded):
        """Process yielded output from transition."""
        # Get output destinations for this transition
        output_dests = self.incidence_matrix.output_destinations.get(trans_idx, {})

        if isinstance(yielded, dict):
            await self._process_dict_yield(trans_idx, yielded, output_dests)
        else:
            await self._process_single_yield(trans_idx, yielded, output_dests)

    async def _process_dict_yield(self, trans_idx: int, yielded: dict, output_dests: Dict[int, int]):
        """Process dictionary yield."""
        wildcard_token = yielded.get('*')
        explicit_targets = set()

        for key, token in yielded.items():
            if key == '*':
                continue

            # Resolve place reference
            place_idx = self._resolve_place_ref(key, output_dests, trans_idx)
            if place_idx is not None:
                explicit_targets.add(place_idx)
                await self._add_token_to_place(trans_idx, place_idx, token)

        if wildcard_token is not None:
            await self._expand_wildcard_to_outputs(trans_idx, wildcard_token, explicit_targets, output_dests)

    async def _process_single_yield(self, trans_idx: int, yielded, output_dests: Dict[int, int]):
        """Process single (place_ref, token) yield."""
        if isinstance(yielded, tuple) and len(yielded) == 2:
            place_ref, token = yielded
            place_idx = self._resolve_place_ref(place_ref, output_dests, trans_idx)
            if place_idx is not None:
                await self._add_token_to_place(trans_idx, place_idx, token)

    def _resolve_place_ref(self, place_ref, output_dests: Dict[int, int], trans_idx: int) -> Optional[int]:
        """Resolve a place reference to a place index.

        Uses parent-relative mapping for subnet instances, similar to NetRuntime.

        Resolution strategies:
        1. Exact match using PlaceRef.get_parts()
        2. Parent-relative mapping using transition's FQN context
        """
        # Extract local_name from place_ref
        if hasattr(place_ref, 'local_name'):
            local_name = place_ref.local_name
        elif isinstance(place_ref, str):
            local_name = place_ref.split('.')[-1]
        else:
            local_name = str(place_ref)

        # Strategy 1: Exact match
        if hasattr(place_ref, 'get_parts'):
            parts = tuple(place_ref.get_parts())
            place_idx = self.incidence_matrix.place_to_idx.get(parts)
            if place_idx is not None:
                return place_idx

        # Strategy 2: Parent-relative mapping (for subnet instances)
        # Get the transition's FQN and extract parent prefix
        trans_fqn = self.incidence_matrix.idx_to_transition.get(trans_idx)
        if trans_fqn and len(trans_fqn) > 1:
            parent_prefix = trans_fqn[:-1]  # Everything except the transition name
            candidate = tuple(list(parent_prefix) + [local_name])
            place_idx = self.incidence_matrix.place_to_idx.get(candidate)
            if place_idx is not None:
                return place_idx

        # Strategy 3: Try as direct string
        if isinstance(place_ref, str):
            parts = tuple(place_ref.split('.'))
            place_idx = self.incidence_matrix.place_to_idx.get(parts)
            if place_idx is not None:
                return place_idx

        # Strategy 4: Handle direct place indices
        if isinstance(place_ref, int) and place_idx in output_dests:
            return place_idx

        return None

    async def _add_token_to_place(self, trans_idx: int, place_idx: int, token: Any):
        """Add a token to a place, handling IO output."""
        place_spec = self._place_specs.get(place_idx)

        if place_spec and place_spec.is_io_output:
            await self._handle_io_output(place_idx, token)
        else:
            self.add_token(place_idx, token)

    async def _expand_wildcard_to_outputs(
        self, trans_idx: int, wildcard_token: Any, explicit_targets: set, output_dests: Dict[int, int]
    ):
        """Expand wildcard token to all output places."""
        for place_idx in output_dests.keys():
            if place_idx not in explicit_targets:
                await self._add_token_to_place(trans_idx, place_idx, wildcard_token)

    async def _fire_transitions(self, firing_vector: List[int]):
        """Fire transitions using matrix multiplication and process data flow.

        This is the CRITICAL method: state transformation is ALWAYS via matmul.

        CRITICAL: When multiple transitions compete for the same input place, we must
        fire them sequentially (each consumes its tokens before the next fires) to ensure
        each token is consumed by exactly one transition per cycle.

        Args:
            firing_vector: Binary vector indicating which transitions fire
        """
        # Step 1: Compute state change via matrix multiplication
        # M_new = M + (A @ F)
        # Note: We compute this for theoretical completeness, but actual state
        # changes happen through token data flow in Step 2
        _state_change = self.incidence_matrix.compute_state_change(firing_vector)

        # Step 2: Fire transitions one at a time, removing tokens immediately
        # This prevents multiple transitions from consuming the same token
        for trans_idx in range(len(firing_vector)):
            if firing_vector[trans_idx] == 0:
                continue

            # Get token consumption slots (handles multiple arcs from same place)
            token_slots = self.incidence_matrix.get_token_slots(trans_idx)
            if not token_slots:
                # No input requirements (generator transition)
                consumed_data = []
                await self._execute_transition(trans_idx, consumed_data)
                self._enabled_times.pop(trans_idx, None)
                continue

            # Collect input tokens by peeking (not removing yet)
            # Group by place to avoid duplicates when multiple slots from same place
            consumed_by_place: Dict[int, List[int]] = {}
            for place_idx, slot_idx in token_slots:
                if place_idx not in consumed_by_place:
                    # Peek at tokens from this place
                    count = self.incidence_matrix.input_requirements[trans_idx][place_idx]
                    tokens = self.marking.peek_tokens(place_idx, count)
                    if len(tokens) < count:
                        # Not enough tokens - skip this transition
                        # (can happen when another transition in same cycle consumed them)
                        consumed_by_place = {}
                        break
                    consumed_by_place[place_idx] = tokens

            # Check if we got all required tokens
            if not consumed_by_place:
                self._enabled_times.pop(trans_idx, None)
                continue

            # Flatten consumed tokens (including duplicates for multiple arcs from same place)
            consumed = []
            for place_idx, slot_idx in token_slots:
                if place_idx in consumed_by_place and consumed_by_place[place_idx]:
                    # Take the next token from this place (repeating if needed for multiple slots)
                    token_idx = slot_idx % len(consumed_by_place[place_idx])
                    consumed.append(consumed_by_place[place_idx][token_idx])

            # Remove consumed tokens IMMEDIATELY before executing
            # This ensures subsequent transitions don't see these tokens
            # Use fast removal for each place
            for place_idx, count in self.incidence_matrix.input_requirements.get(trans_idx, {}).items():
                if place_idx in consumed_by_place:
                    # Get the tokens from this place (consumed_by_place has the original order)
                    place_tokens_to_remove = consumed_by_place[place_idx][:count]
                    # Use optimized removal
                    self.marking.remove_tokens_fast(place_idx, place_tokens_to_remove)

            # Convert token IDs to data
            consumed_data = [self.token_registry.get(tid) for tid in consumed]

            # Execute transition
            await self._execute_transition(trans_idx, consumed_data)

            # Clear enabled time after firing
            self._enabled_times.pop(trans_idx, None)

    async def run_cycle(self):
        """Run a single execution cycle."""
        # Build firing vector
        firing_vector = await self._build_firing_vector()

        # Fire transitions (matmul + data flow)
        await self._fire_transitions(firing_vector)

        # Return whether any transitions fired
        return any(firing_vector)

    async def run(self):
        """Main execution loop."""
        await self._start_io_input()

        try:
            cycle_timeout = 30.0  # Safety timeout for single cycle (seconds)

            # For pure computational nets (no async features), use minimal sleeps
            use_fast_loop = not self._has_async_features

            while not self._stop_event.is_set():
                # Record cycle start time BEFORE running the cycle
                cycle_start = time.perf_counter()

                fired = await self.run_cycle()

                # Check for hard loop (single cycle taking too long)
                cycle_elapsed = time.perf_counter() - cycle_start
                if cycle_elapsed > cycle_timeout:
                    logger.warning(" Cycle timeout detected (%.2fs), stopping to prevent hard loop",
                                 cycle_elapsed)
                    break
                await asyncio.sleep(1e-10)

        except asyncio.CancelledError:
            logger.debug(" Execution cancelled")
            pass
        except Exception as e:
            logger.error(" Error in execution loop: %s", e)
            raise
        finally:
            # Cleanup IO tasks - cancel all and wait for completion
            logger.debug(" Cleaning up %d IO tasks", len(self._io_tasks))
            for task in self._io_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete (with return_exceptions to avoid CancelledError propagation)
            if self._io_tasks:
                await asyncio.gather(*self._io_tasks, return_exceptions=True)

            logger.debug(" Cleanup complete")

    async def start(self):
        """Start the runtime."""
        self._stop_event.clear()
        # Create the run task
        task = asyncio.create_task(self.run())
        # Give the run loop a chance to start by yielding control
        await asyncio.sleep(0)
        # Store the task so we can cancel it later if needed
        self._run_task = task

    async def stop(self, timeout: float = 5.0):
        """Stop the runtime.

        Args:
            timeout: Maximum time to wait for graceful shutdown (seconds)
        """
        logger.debug(" Stopping runtime (timeout=%.2fs)", timeout)

        # Set stop event to signal run loop to exit
        self._stop_event.set()

        # Wait for the run task to finish (with timeout)
        # The run() method handles IO task cleanup in its finally block
        if self._run_task and not self._run_task.done():
            try:
                await asyncio.wait_for(self._run_task, timeout=timeout)
                logger.debug(" Run task completed gracefully")
            except asyncio.TimeoutError:
                logger.warning(" Run task did not complete in %.2fs, cancelling", timeout)
                self._run_task.cancel()
                try:
                    await self._run_task
                    logger.debug(" Run task cancelled successfully")
                except asyncio.CancelledError:
                    logger.debug(" Run task cancelled with CancelledError")
                except Exception as e:
                    logger.error(" Error cancelling run task: %s", e)
            except Exception as e:
                logger.error(" Error waiting for run task: %s", e)

        logger.debug(" Stop complete")

    # =============================================================================
    # Synchronous Execution Mode
    # =============================================================================

    def _build_firing_vector_sync(self) -> List[int]:
        """Build firing vector F by evaluating enabled transitions (synchronous).

        Returns:
            List of 0/1 values indicating which transitions fire
        """
        firing_vector = [0] * self.incidence_matrix.num_transitions

        # Get count dict once per cycle (cached in Marking)
        count_dict = self.marking.get_count_dict()

        for trans_idx in range(self.incidence_matrix.num_transitions):
            # Check if enabled (has sufficient tokens)
            if not self.incidence_matrix.is_enabled(trans_idx, count_dict):
                self._enabled_times.pop(trans_idx, None)
                continue

            # Track enabled time for delay support
            trans_spec = self._transition_specs.get(trans_idx)
            if trans_spec and trans_spec.delay > 0:
                if trans_idx not in self._enabled_times:
                    self._enabled_times[trans_idx] = self.timebase.now()

                # Check if delay has elapsed
                enabled_time = self._enabled_times.get(trans_idx)
                if enabled_time is None:
                    continue

                elapsed = self.timebase.now() - enabled_time
                if elapsed < trans_spec.delay:
                    continue

            # Evaluate guard (synchronous only - async guards not supported in sync mode)
            guard_result = self.guard_matrix.evaluate(
                trans_idx,
                self.marking,
                self.incidence_matrix,
                self.bb,
                self.timebase,
                self.token_registry,
            )

            # Skip async guards in sync mode
            if inspect.isasyncgen(guard_result):
                continue
            elif isinstance(guard_result, bool) and not guard_result:
                continue
            elif not guard_result:
                continue

            # Transition fires!
            firing_vector[trans_idx] = 1

        return firing_vector

    def _execute_transition_sync(self, trans_idx: int, input_tokens: List[Any]):
        """Execute a transition with input tokens (synchronous).

        Args:
            trans_idx: Transition index
            input_tokens: List of consumed token data
        """
        trans_spec = self._transition_specs.get(trans_idx)
        if not trans_spec:
            return

        # Skip async transitions in sync mode
        if inspect.iscoroutinefunction(trans_spec.handler):
            return

        # Create interface view if needed
        bb_to_pass = self._create_interface_view_if_needed(self.bb, trans_spec.handler)

        # Execute transition
        trans_state = self._transition_states.get(trans_idx)

        if trans_state is not None:
            logger.debug("[matrix_fire_sync] trans=%s consumed=%s",
                        self.incidence_matrix.idx_to_transition[trans_idx], input_tokens)
            results = trans_spec.handler(input_tokens, bb_to_pass, self.timebase, trans_state)
        else:
            logger.debug("[matrix_fire_sync] trans=%s consumed=%s",
                        self.incidence_matrix.idx_to_transition[trans_idx], input_tokens)
            results = trans_spec.handler(input_tokens, bb_to_pass, self.timebase)

        # Process results (synchronous generators only)
        if inspect.isgenerator(results):
            for yielded in results:
                self._process_yield_sync(trans_idx, yielded)
        elif results is not None:
            self._process_yield_sync(trans_idx, results)

    def _process_yield_sync(self, trans_idx: int, yielded):
        """Process yielded output from transition (synchronous)."""
        # Get output destinations for this transition
        output_dests = self.incidence_matrix.output_destinations.get(trans_idx, {})

        if isinstance(yielded, dict):
            self._process_dict_yield_sync(trans_idx, yielded, output_dests)
        else:
            self._process_single_yield_sync(trans_idx, yielded, output_dests)

    def _process_dict_yield_sync(self, trans_idx: int, yielded: dict, output_dests: Dict[int, int]):
        """Process dictionary yield (synchronous)."""
        wildcard_token = yielded.get('*')
        explicit_targets = set()

        for key, token in yielded.items():
            if key == '*':
                continue

            # Resolve place reference
            place_idx = self._resolve_place_ref(key, output_dests, trans_idx)
            if place_idx is not None:
                explicit_targets.add(place_idx)
                self._add_token_to_place_sync(trans_idx, place_idx, token)

        if wildcard_token is not None:
            self._expand_wildcard_to_outputs_sync(trans_idx, wildcard_token, explicit_targets, output_dests)

    def _process_single_yield_sync(self, trans_idx: int, yielded, output_dests: Dict[int, int]):
        """Process single (place_ref, token) yield (synchronous)."""
        if isinstance(yielded, tuple) and len(yielded) == 2:
            place_ref, token = yielded
            place_idx = self._resolve_place_ref(place_ref, output_dests, trans_idx)
            if place_idx is not None:
                self._add_token_to_place_sync(trans_idx, place_idx, token)

    def _add_token_to_place_sync(self, trans_idx: int, place_idx: int, token: Any):
        """Add a token to a place, handling IO output (synchronous).

        Note: IO output places are not supported in sync mode - tokens are just added.
        """
        # Skip IO output in sync mode (just add token to marking)
        self.add_token(place_idx, token)

    def _expand_wildcard_to_outputs_sync(
        self, trans_idx: int, wildcard_token: Any, explicit_targets: set, output_dests: Dict[int, int]
    ):
        """Expand wildcard token to all output places (synchronous)."""
        for place_idx in output_dests.keys():
            if place_idx not in explicit_targets:
                self._add_token_to_place_sync(trans_idx, place_idx, wildcard_token)

    def _fire_transitions_sync(self, firing_vector: List[int]):
        """Fire transitions using matrix multiplication and process data flow (synchronous).

        This is the synchronous version of _fire_transitions for pure computational nets.

        CRITICAL: When multiple transitions compete for the same input place, we must
        fire them sequentially (each consumes its tokens before the next fires) to ensure
        each token is consumed by exactly one transition per cycle.

        Args:
            firing_vector: Binary vector indicating which transitions fire
        """
        # Step 1: Compute state change via matrix multiplication
        # M_new = M + (A @ F)
        _state_change = self.incidence_matrix.compute_state_change(firing_vector)

        # Step 2: Fire transitions one at a time, removing tokens immediately
        # This prevents multiple transitions from consuming the same token
        for trans_idx in range(len(firing_vector)):
            if firing_vector[trans_idx] == 0:
                continue

            # Get token consumption slots (handles multiple arcs from same place)
            token_slots = self.incidence_matrix.get_token_slots(trans_idx)
            if not token_slots:
                # No input requirements (generator transition)
                consumed_data = []
                self._execute_transition_sync(trans_idx, consumed_data)
                self._enabled_times.pop(trans_idx, None)
                continue

            # Collect input tokens by peeking (not removing yet)
            # Group by place to avoid duplicates when multiple slots from same place
            consumed_by_place: Dict[int, List[int]] = {}
            for place_idx, slot_idx in token_slots:
                if place_idx not in consumed_by_place:
                    # Peek at tokens from this place
                    count = self.incidence_matrix.input_requirements[trans_idx][place_idx]
                    tokens = self.marking.peek_tokens(place_idx, count)
                    if len(tokens) < count:
                        # Not enough tokens - skip this transition
                        # (can happen when another transition in same cycle consumed them)
                        consumed_by_place = {}
                        break
                    consumed_by_place[place_idx] = tokens

            # Check if we got all required tokens
            if not consumed_by_place:
                self._enabled_times.pop(trans_idx, None)
                continue

            # Flatten consumed tokens (including duplicates for multiple arcs from same place)
            consumed = []
            for place_idx, slot_idx in token_slots:
                if place_idx in consumed_by_place and consumed_by_place[place_idx]:
                    # Take the next token from this place (repeating if needed for multiple slots)
                    token_idx = slot_idx % len(consumed_by_place[place_idx])
                    consumed.append(consumed_by_place[place_idx][token_idx])

            # Remove consumed tokens IMMEDIATELY before executing
            # This ensures subsequent transitions don't see these tokens
            # Use fast removal for each place
            for place_idx, count in self.incidence_matrix.input_requirements.get(trans_idx, {}).items():
                if place_idx in consumed_by_place:
                    # Get the tokens from this place (consumed_by_place has the original order)
                    place_tokens_to_remove = consumed_by_place[place_idx][:count]
                    # Use optimized removal
                    self.marking.remove_tokens_fast(place_idx, place_tokens_to_remove)

            # Convert token IDs to data
            consumed_data = [self.token_registry.get(tid) for tid in consumed]

            # Execute transition
            self._execute_transition_sync(trans_idx, consumed_data)

            # Clear enabled time after firing
            self._enabled_times.pop(trans_idx, None)

    def run_cycle_sync(self) -> bool:
        """Run a single execution cycle (synchronous).

        Returns:
            True if any transitions fired, False otherwise
        """
        # Build firing vector
        firing_vector = self._build_firing_vector_sync()

        # Fire transitions (matmul + data flow)
        self._fire_transitions_sync(firing_vector)

        # Return whether any transitions fired
        return any(firing_vector)

    def run_sync(self, max_cycles: int = 100000):
        """Run the net to completion or max cycles (synchronous execution).

        This is the high-performance synchronous execution path for pure
        computational nets. It avoids all asyncio overhead.

        Args:
            max_cycles: Maximum number of cycles to execute (prevents infinite loops)

        Returns:
            Number of cycles executed

        Raises:
            RuntimeError: If net has async features (use async mode instead)
        """
        if self._has_async_features:
            raise RuntimeError(
                "Net has async features (IO places, async transitions/guards). "
                "Use async mode (run/start) instead."
            )

        cycles = 0
        start_time = time.perf_counter()
        cycle_timeout = 10.0  # Safety timeout for single cycle (seconds)

        for cycles in range(max_cycles):
            # Check stop flag at start of each cycle
            if self._stop_event_sync:
                logger.debug("[run_sync] Stopped by stop flag after %d cycles", cycles)
                break

            # Record cycle start time BEFORE running the cycle
            cycle_start = time.perf_counter()

            # Run one cycle
            fired = self.run_cycle_sync()

            # Check if single cycle is taking too long (possible hard loop)
            cycle_elapsed = time.perf_counter() - cycle_start
            if cycle_elapsed > cycle_timeout:
                logger.warning("[run_sync] Cycle timeout after %d cycles (single cycle took %.2fs)",
                             cycles, cycle_elapsed)
                break

            # If no transitions fired, we're done
            if not fired:
                break

            # Check stop flag again after firing (in case a transition set it)
            if self._stop_event_sync:
                logger.debug("[run_sync] Stopped by stop flag after firing in cycle %d", cycles)
                break

        elapsed = time.perf_counter() - start_time
        logger.debug("[run_sync] Completed %d cycles in %.4fms (%.0f cycles/sec)",
                     cycles, elapsed * 1000, cycles / elapsed if elapsed > 0 else 0)

        return cycles

    def stop_sync(self):
        """Stop the synchronous execution loop."""
        self._stop_event_sync = True


class Runner:
    """High-level runner for executing a Petri net

    Uses MatrixRuntime for high-performance matrix-based execution.
    """

    def __init__(self, net_func: Any, blackboard: Any):
        if not hasattr(net_func, '_spec'):
            raise ValueError(f"{net_func} is not a valid net")

        self.spec = net_func._spec
        self.blackboard = blackboard
        self.timebase = None
        self._matrix_runtime: Optional[MatrixRuntime] = None

    @property
    def runtime(self):
        """Get the active runtime (MatrixRuntime)."""
        return self._matrix_runtime

    async def start(self, timebase: Any):
        """Start the net with given timebase"""
        self.timebase = timebase

        self._matrix_runtime = MatrixRuntime(
            self.spec,
            self.blackboard,
            self.timebase
        )
        await self._matrix_runtime.start()

    async def stop(self, timeout: float = 5.0):
        """Stop the net"""
        if self._matrix_runtime:
            await self._matrix_runtime.stop(timeout)

    def add_place(self, fqn: str,
                  state_factory: Optional[Callable] = None):
        """Add a place to the running net"""
        if not self.runtime:
            raise RuntimeError("Net is not running. Call start() first.")
        return self.runtime.add_place(fqn, state_factory)

    def add_transition(self, fqn: str, handler: Callable,
                      guard: Optional[GuardSpec] = None,
                      state_factory: Optional[Callable] = None,
                      delay: float = 0.0):
        """Add a transition to the running net"""
        if not self.runtime:
            raise RuntimeError("Net is not running. Call start() first.")
        return self.runtime.add_transition(fqn, handler, guard, state_factory, delay)

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

    def run_sync(self, timebase: Any, max_cycles: int = 100000) -> int:
        """Run the net to completion in synchronous mode.

        This is the high-performance execution path for pure computational nets.
        Only works with MatrixRuntime and requires no async features.

        Args:
            timebase: Timebase for timing operations
            max_cycles: Maximum number of cycles to execute

        Returns:
            Number of cycles executed

        Raises:
            RuntimeError: If net has async features
        """
        self.timebase = timebase

        self._matrix_runtime = MatrixRuntime(
            self.spec,
            self.blackboard,
            self.timebase
        )
        return self._matrix_runtime.run_sync(max_cycles)
