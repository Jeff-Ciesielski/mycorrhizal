#!/usr/bin/env python3
"""
Hypha Bridge - Wrapper/decorator system for BT-in-PN integration.

This module extends Hypha's API to support BT-in-PN integration without modifying
the core Hypha modules (specs.py, builder.py, runtime.py).

Uses a global registry to store metadata about BT-enabled transitions,
then wraps the Runner to intercept transition firing and execute BTs.
"""

from typing import Any, Callable, List, Optional, Dict
from functools import wraps
import asyncio

from ..hypha.core.specs import PlaceRef, TransitionRef
from ..rhizomorph.core import Status
from .tree_spec import PNIntegration
from .exceptions import MyceliumError


# Global registry for PN integrations
# Key: (net_name, transition_name) -> PNIntegration
_pn_integration_registry: Dict[tuple[str, str], PNIntegration] = {}

# Global set to track which transitions are currently processing batches (to prevent duplicate processing)
_batch_processing_transitions: set = set()


def register_pn_integration(net_name: str, transition_name: str, integration: PNIntegration) -> None:
    """Register a PN integration for a transition."""
    key = (net_name, transition_name)
    _pn_integration_registry[key] = integration


def get_pn_integration(net_name: str, transition_name: str) -> Optional[PNIntegration]:
    """Get PN integration for a transition."""
    key = (net_name, transition_name)
    return _pn_integration_registry.get(key)


def clear_pn_integration_registry() -> None:
    """Clear the PN integration registry (useful for testing)."""
    global _pn_integration_registry
    _pn_integration_registry = {}


class MyceliumNetBuilder:
    """
    Wrapper around Hypha's NetBuilder that adds BT-in-PN support.

    Provides the same API as NetBuilder but intercepts transition() calls
    to store BT integration metadata.
    """

    def __init__(self, net_builder: Any, net_name: str):
        """
        Initialize the wrapper.

        Args:
            net_builder: The original Hypha NetBuilder
            net_name: Name of the net (for registry key)
        """
        self._builder = net_builder
        self._net_name = net_name

    def place(self, *args, **kwargs):
        """Forward to original builder."""
        return self._builder.place(*args, **kwargs)

    def io_input_place(self, *args, **kwargs):
        """Forward to original builder."""
        return self._builder.io_input_place(*args, **kwargs)

    def io_output_place(self, *args, **kwargs):
        """Forward to original builder."""
        return self._builder.io_output_place(*args, **kwargs)

    def guard(self, *args, **kwargs):
        """Forward to original builder."""
        return self._builder.guard(*args, **kwargs)

    def subnet(self, *args, **kwargs):
        """Forward to original builder."""
        return self._builder.subnet(*args, **kwargs)

    def transition(self, bt=None, fsm=None, bt_mode="token", outputs=None, **kwargs):
        """
        Decorator for transitions with optional BT or FSM integration.

        Args:
            bt: Optional BT tree to integrate into this transition
            fsm: Optional FSM initial state to integrate into this transition
            bt_mode: "token" (process one token) or "batch" (process all tokens) - BT only
            outputs: List of output PlaceRef objects (required when bt or fsm is provided)
            **kwargs: Additional arguments passed to original builder.transition()
        """
        # Validate that only one integration type is specified
        if bt is not None and fsm is not None:
            raise ValueError(
                f"Transition cannot have both BT and FSM integration. "
                f"Specify either 'bt' or 'fsm', not both."
            )

        # Validate BT integration parameters
        if bt is not None:
            if not outputs:
                raise ValueError(
                    f"Transition with BT integration must specify 'outputs' parameter "
                    f"(list of output PlaceRef objects)"
                )
            if bt_mode not in ("token", "batch"):
                raise ValueError(
                    f"Invalid bt_mode: {bt_mode}. Must be 'token' or 'batch'"
                )

        # Validate FSM integration parameters
        if fsm is not None:
            if not outputs:
                raise ValueError(
                    f"Transition with FSM integration must specify 'outputs' parameter "
                    f"(list of output PlaceRef objects)"
                )

        # Get the original transition decorator
        original_decorator = self._builder.transition(**kwargs)

        def decorator(func: Callable) -> TransitionRef:
            # Apply the original decorator to get the TransitionRef
            transition_ref = original_decorator(func)

            # If BT integration is specified, register it
            if bt is not None:
                transition_name = transition_ref.local_name

                integration = PNIntegration(
                    transition_name=transition_name,
                    integration_type="bt",
                    mode=bt_mode,
                    bt_tree=bt,
                    output_places=outputs or [],
                )

                register_pn_integration(self._net_name, transition_name, integration)

                # Store metadata on the transition function for runtime access
                func._mycelium_bt_integration = integration  # type: ignore[attr-defined]

            # If FSM integration is specified, register it
            if fsm is not None:
                transition_name = transition_ref.local_name

                integration = PNIntegration(
                    transition_name=transition_name,
                    integration_type="fsm",
                    initial_state=fsm,
                    output_places=outputs or [],
                )

                register_pn_integration(self._net_name, transition_name, integration)

                # Store metadata on the transition function for runtime access
                func._mycelium_fsm_integration = integration  # type: ignore[attr-defined]

            return transition_ref

        return decorator

    def arc(self, *args, **kwargs):
        """Forward to original builder."""
        return self._builder.arc(*args, **kwargs)

    def __getattr__(self, name: str):
        """Forward any other attributes to the original builder."""
        return getattr(self._builder, name)


def create_net_wrapper(net_func: Callable, net_name: str) -> tuple[Any, MyceliumNetBuilder]:
    """
    Create a wrapped net builder that supports BT-in-PN integration.

    Called by the @pn_net decorator to wrap the Hypha NetBuilder.

    Args:
        net_func: The net definition function
        net_name: Name of the net

    Returns:
        Tuple of (original NetSpec, wrapped MyceliumNetBuilder)
    """
    from ..hypha.core.builder import NetBuilder

    # Create the original builder
    original_builder = NetBuilder(net_name)

    # Wrap it with our MyceliumNetBuilder
    wrapped_builder = MyceliumNetBuilder(original_builder, net_name)

    # Call the net function with the wrapped builder
    net_func(wrapped_builder)

    # Return the NetSpec from the original builder
    return original_builder.spec, wrapped_builder


class TransitionRuntimeWrapper:
    """
    Wrapper for TransitionRuntime that adds BT execution support.

    Wraps the original transition handler to check for BT integration,
    execute the BT, and use its routing decisions.
    """

    def __init__(self, original_handler: Callable, pn_integration: Optional[PNIntegration], transition_ref=None):
        """
        Initialize the wrapper.

        Args:
            original_handler: The original transition handler function
            pn_integration: PN integration metadata (if any)
            transition_ref: Reference to the transition runtime for returning rejected tokens
        """
        self._original_handler = original_handler
        self._pn_integration = pn_integration
        self._transition_ref = transition_ref
        self._batch_processing = False  # Flag to prevent duplicate batch processing
        self._batch_lock = asyncio.Lock()  # Lock for atomic batch processing

    async def __call__(self, consumed, bb, timebase, state=None):
        """
        Execute the transition with BT integration if configured.

        Args:
            consumed: List of tokens (flattened from all input places)
            bb: Blackboard
            timebase: Timebase
            state: Optional transition state

        Returns:
            Yields of output tokens (like a normal transition)
        """
        import logging
        logger = logging.getLogger(__name__)

        from .pn_context import PNContext
        from ..rhizomorph.core import Runner as BTRunner, Status

        # consumed is already a flattened list of tokens
        tokens = consumed if isinstance(consumed, list) else [consumed]

        logger.info(f"TransitionRuntimeWrapper called with {len(tokens)} tokens, pn_integration={self._pn_integration}, tokens={[t for t in tokens]}")

        # If no integration, or FSM integration (not yet implemented), execute original handler
        if self._pn_integration is None or self._pn_integration.integration_type == "fsm":
            # FSM integration: visualization only, execution is manual in transition handler
            if self._pn_integration and self._pn_integration.integration_type == "fsm":
                logger.info(f"FSM integration detected (visualization only), executing original handler")

            async def gen():
                if state is not None:
                    async for result in self._original_handler(consumed, bb, timebase, state):
                        yield result
                else:
                    async for result in self._original_handler(consumed, bb, timebase):
                        yield result

            async for result in gen():
                yield result
            return

        # BT integration: Execute BT and use its routing decisions
        integration = self._pn_integration
        bt_tree = integration.bt_tree
        mode = integration.mode
        output_places = integration.output_places

        # Get net runtime from blackboard (injected by PN runner)
        net_runtime = getattr(bb, '_mycelium_net_runtime', None)
        if net_runtime is None:
            raise RuntimeError(
                f"Transition has BT integration but no net runtime available on blackboard. "
                f"Did you use mycorrhizal.mycelium.Runner instead of mycorrhizal.hypha.core.Runner?"
            )

        # Determine which tokens to process
        if mode == "token":
            # Track token origins by scanning input places before processing
            # Needed to properly return rejected tokens to their source
            # Use id(token) as key because tokens (Pydantic models) aren't hashable
            token_origins = {}
            if self._transition_ref and self._transition_ref.input_arcs:
                for place_parts, _ in self._transition_ref.input_arcs:
                    place = net_runtime.places.get(place_parts)
                    if place and place.tokens:
                        for token in place.tokens:
                            token_origins[id(token)] = (token, place_parts)

            logger.info(f"Processing {len(tokens)} tokens in token mode")
            for i, token in enumerate(tokens):
                logger.info(f"Processing token {i+1}/{len(tokens)}: {token}")
                await self._execute_bt_for_token(
                    bt_tree, token, bb, timebase, net_runtime, output_places, token_origins
                )

            # After processing tokens, check if more tokens remain in input places
            # If so, set their token_added_event to wake the transition again
            # Workaround for Hypha's event-driven runtime which clears the event before firing
            for transition in net_runtime.transitions.values():
                for place_parts, _ in transition.input_arcs:
                    place = net_runtime.places.get(place_parts)
                    if place and len(place.tokens) > 0:
                        # Tokens remain, set event to wake transition again
                        place.token_added_event.set()
                        logger.info(f"[BRIDGE] Set token_added_event for {place_parts} with {len(place.tokens)} tokens remaining")
        else:  # batch mode
            # Batch mode limitation: Transitions fire immediately when tokens become available,
            # leading to multiple firings before all tokens are accumulated.
            # Works best when all tokens are added to input places before the net starts running.
            # For most use cases, token mode (the default) is recommended.

            # Create a unique key for this transition
            transition_key = (id(net_runtime), id(self._transition_ref)) if self._transition_ref else (id(net_runtime), id(self))

            # Check if we're already processing this batch
            if transition_key in _batch_processing_transitions:
                logger.info(f"Batch mode: already processing {transition_key}, skipping duplicate call")
                if False:
                    yield
                return

            # Mark this transition as processing
            _batch_processing_transitions.add(transition_key)
            logger.info(f"Batch mode: starting processing {transition_key}")

            try:
                all_tokens = list(tokens)  # Start with tokens from this firing

                # First, consume ALL tokens from input places
                if self._transition_ref and self._transition_ref.input_arcs:
                    for place_parts, _ in self._transition_ref.input_arcs:
                        place = net_runtime.places.get(place_parts)
                        if place and place.tokens:
                            # Consume all remaining tokens from this place
                            remaining = list(place.tokens)
                            place.remove_tokens(remaining)
                            all_tokens.extend(remaining)
                            logger.info(f"Batch mode: consumed {len(remaining)} more tokens from {place_parts}")

                # Remove duplicates while preserving order
                seen = set()
                unique_tokens = []
                for token in all_tokens:
                    if token not in seen:
                        seen.add(token)
                        unique_tokens.append(token)

                logger.info(f"Batch mode: processing {len(unique_tokens)} unique tokens")

                await self._execute_bt_for_batch(
                    bt_tree, unique_tokens, bb, timebase, net_runtime, output_places
                )
            finally:
                # Remove from processing set
                _batch_processing_transitions.discard(transition_key)
                logger.info(f"Batch mode: done processing {transition_key}")

        # Yield nothing - BT has already routed tokens directly to output places
        # Must be an async generator, so check if there are more results
        # The `yield` below makes this function an async generator
        # Will only be reached if the loop above doesn't process all tokens
        if False:
            yield

    async def _execute_bt_for_token(
        self, bt_tree, token, bb, timebase, net_runtime, output_places, token_origins
    ):
        """Execute BT for a single token."""
        import logging
        logger = logging.getLogger(__name__)

        from .pn_context import PNContext
        from ..rhizomorph.core import Runner as BTRunner, Status

        # Create PNContext with token origin tracking
        # token_origins maps id(token) -> (token, place_parts)
        pn_ctx = PNContext(
            net_runtime=net_runtime,
            output_places=output_places,
            tokens=[token],
            timebase=timebase,
            token_origins=token_origins,
        )

        # Inject PNContext into blackboard
        # Use setattr with extra='allow' for Pydantic models
        try:
            bb.pn_ctx = pn_ctx
        except Exception:
            # If setattr fails, store in dict for cleanup
            if not hasattr(bb, '_pn_ctx_dict'):
                bb._pn_ctx_dict = {}
            bb._pn_ctx_dict['pn_ctx'] = pn_ctx

        try:
            # Create BT runner (fresh runner for each execution to avoid state issues)
            bt_runner = BTRunner(bt_tree, bb=bb, tb=timebase)

            # Run BT to completion
            max_ticks = 100  # Safety limit
            ticks = 0
            result = Status.RUNNING

            while result == Status.RUNNING and ticks < max_ticks:
                result = await bt_runner.tick()
                ticks += 1

            # Execute routing decisions made by the BT
            routing_decisions = pn_ctx.get_routing_decisions()
            rejected_tokens = pn_ctx.get_rejected_tokens()
            deferred_tokens = pn_ctx.get_deferred_tokens()

            print(f"[BRIDGE] BT completed with {len(routing_decisions)} routing decisions")

            # Apply routing: add tokens to output places
            for routed_token, place_ref in routing_decisions:
                # Resolve place_ref to place runtime
                place_parts = tuple(place_ref.get_parts())  # Convert to tuple
                print(f"[BRIDGE] Routing token to place: {place_parts}")
                if place_parts in net_runtime.places:
                    place_runtime = net_runtime.places[place_parts]
                    place_runtime.add_token(routed_token)
                    # Set token_added_event immediately after adding token
                    # Ensures dependent transitions wake up and process the token
                    place_runtime.token_added_event.set()
                    print(f"[BRIDGE] Token added successfully, place now has {len(place_runtime.tokens)} tokens")
                else:
                    print(f"[BRIDGE] Place not found: {place_parts}")

            # Handle rejected tokens - return them to their input places
            if rejected_tokens:
                for rejected_token, reason in rejected_tokens:
                    # Get the token's origin from PNContext
                    # token_origins maps id(token) -> (token, place_parts)
                    origin_parts = pn_ctx.get_token_origin(rejected_token)

                    # Fallback to first input place if origin not tracked
                    if origin_parts is None and self._transition_ref and self._transition_ref.input_arcs:
                        origin_parts = self._transition_ref.input_arcs[0][0]

                    if origin_parts:
                        input_place = net_runtime.places.get(origin_parts)
                        if input_place:
                            input_place.add_token(rejected_token)
                            logger.info(f"Returned rejected token to {origin_parts}: {reason}")
                        else:
                            logger.warning(f"Could not find origin place {origin_parts} for rejected token")
                    else:
                        logger.warning(f"No origin place found for rejected token (id={id(rejected_token)}): {reason}")

            # Deferred tokens are implicitly handled by not removing them from input
            # (they're already back in the input places)

            if not routing_decisions and not rejected_tokens and not deferred_tokens:
                # No routing decisions - might be an error
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"BT completed without making routing decisions. "
                    f"Token will be lost."
                )

        finally:
            # Clean up PNContext
            if hasattr(bb, 'pn_ctx'):
                delattr(bb, 'pn_ctx')
            if hasattr(bb, '_pn_ctx_dict'):
                bb._pn_ctx_dict.pop('pn_ctx', None)

    async def _execute_bt_for_batch(
        self, bt_tree, tokens, bb, timebase, net_runtime, output_places
    ):
        """Execute BT for all tokens in batch mode."""
        from .pn_context import PNContext
        from ..rhizomorph.core import Runner as BTRunner

        # Create PNContext with all tokens
        pn_ctx = PNContext(
            net_runtime=net_runtime,
            output_places=output_places,
            tokens=tokens,
            timebase=timebase,
        )

        # Inject PNContext into blackboard
        # Use setattr with extra='allow' for Pydantic models
        try:
            bb.pn_ctx = pn_ctx
        except Exception:
            # If setattr fails, store in dict for cleanup
            if not hasattr(bb, '_pn_ctx_dict'):
                bb._pn_ctx_dict = {}
            bb._pn_ctx_dict['pn_ctx'] = pn_ctx

        try:
            # Create BT runner
            bt_runner = BTRunner(bt_tree, bb=bb, tb=timebase)

            # Run BT to completion
            max_ticks = 100
            ticks = 0
            result = Status.RUNNING

            while result == Status.RUNNING and ticks < max_ticks:
                result = await bt_runner.tick()
                ticks += 1

            # Execute routing decisions made by the BT
            routing_decisions = pn_ctx.get_routing_decisions()
            rejected_tokens = pn_ctx.get_rejected_tokens()
            deferred_tokens = pn_ctx.get_deferred_tokens()

            print(f"[BRIDGE] BT completed with {len(routing_decisions)} routing decisions")

            # Apply routing: add tokens to output places
            for routed_token, place_ref in routing_decisions:
                # Resolve place_ref to place runtime
                place_parts = tuple(place_ref.get_parts())  # Convert to tuple
                print(f"[BRIDGE] Routing token to place: {place_parts}")
                if place_parts in net_runtime.places:
                    place_runtime = net_runtime.places[place_parts]
                    place_runtime.add_token(routed_token)
                    # Set token_added_event immediately after adding token
                    # Ensures dependent transitions wake up and process the token
                    place_runtime.token_added_event.set()
                    print(f"[BRIDGE] Token added successfully, place now has {len(place_runtime.tokens)} tokens")
                else:
                    print(f"[BRIDGE] Place not found: {place_parts}")

            # Handle rejected tokens
            if rejected_tokens:
                import logging
                logger = logging.getLogger(__name__)
                for rejected_token, reason in rejected_tokens:
                    logger.warning(f"Token rejected: {reason}")

            if not routing_decisions and not rejected_tokens and not deferred_tokens:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"BT completed without making routing decisions. "
                    f"Tokens will be lost."
                )

        finally:
            # Clean up PNContext
            if hasattr(bb, 'pn_ctx'):
                delattr(bb, 'pn_ctx')
            if hasattr(bb, '_pn_ctx_dict'):
                bb._pn_ctx_dict.pop('pn_ctx', None)


class PNRunner:
    """
    Wrapper around Hypha's Runner that adds BT-in-PN and FSM-in-PN support.

    Provides the same API as Runner but injects BT/FSM execution support
    into the NetRuntime.
    """

    def __init__(self, net_func: Any, blackboard: Any):
        """
        Initialize the runner.

        Args:
            net_func: The @pn_net decorated function
            blackboard: The blackboard for shared state
        """
        from ..hypha.core import Runner

        # Create the original runner
        self._runner = Runner(net_func, blackboard)
        self._blackboard = blackboard
        self._timebase = None

        # Store net spec for BT/FSM integration lookup
        self._spec = self._runner.spec
        self._net_name = self._spec.name

    async def start(self, timebase: Any):
        """
        Start the PN runner with BT-in-PN support.

        Args:
            timebase: The timebase for timing operations
        """
        from .hypha_bridge import get_pn_integration, TransitionRuntimeWrapper

        self._timebase = timebase

        # Start the original runner
        await self._runner.start(timebase)

        # Get the runtime
        runtime = self._runner.runtime

        # Inject net runtime into blackboard for BT access
        self._blackboard._mycelium_net_runtime = runtime

        # Wrap transitions with BT integration support
        for trans_fqn, trans_runtime in runtime.transitions.items():
            # Extract transition name from FQN
            # trans_fqn is a tuple like ('BTInPNDemo', 'route_by_priority')
            trans_name = trans_fqn[-1] if trans_fqn else ""

            # Check if this transition has BT integration
            integration = get_pn_integration(self._net_name, trans_name)

            if integration is not None:
                # Wrap the transition handler (stored in spec.handler)
                handler = trans_runtime.spec.handler
                if handler is None:
                    continue

                wrapped_handler = TransitionRuntimeWrapper(handler, integration, transition_ref=trans_runtime)
                trans_runtime.spec.handler = wrapped_handler
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Wrapped transition '{trans_name}' (FQN: {trans_fqn}) with BT integration")

    async def stop(self, timeout: float = 5.0):
        """Stop the runner."""
        # Clean up blackboard
        if hasattr(self._blackboard, '_mycelium_net_runtime'):
            delattr(self._blackboard, '_mycelium_net_runtime')

        await self._runner.stop(timeout)

    @property
    def runtime(self):
        """Get the underlying PN runtime."""
        return self._runner.runtime

    @property
    def spec(self):
        """Get the net specification."""
        return self._runner.spec

    @property
    def timebase(self):
        """Get the timebase."""
        return self._runner.timebase

    def add_place(self, *args, **kwargs):
        """Forward to original runner."""
        return self._runner.add_place(*args, **kwargs)

    def add_transition(self, *args, **kwargs):
        """Forward to original runner."""
        return self._runner.add_transition(*args, **kwargs)

    def add_arc(self, *args, **kwargs):
        """Forward to original runner."""
        return self._runner.add_arc(*args, **kwargs)

    def to_mermaid(self) -> str:
        """
        Generate Mermaid diagram of the net with BT/FSM integration information.

        Extends the standard NetSpec.to_mermaid() to include information about
        which transitions have BT or FSM integrations.

        For FSM integrations, embeds the full FSM state diagram as a subgraph
        within the transition node, showing all states and transitions.
        """
        lines = ["graph TD"]

        def add_subnet(spec, indent: str = "    "):
            spec_fqn = spec.get_fqn()

            if spec.subnets:
                for subnet_name, subnet_spec in spec.subnets.items():
                    subnet_fqn = subnet_spec.get_fqn()
                    lines.append(f"{indent}subgraph {subnet_fqn}")
                    add_subnet(subnet_spec, indent + "    ")
                    lines.append(f"{indent}end")

            for place_name, place_spec in spec.places.items():
                place_fqn = spec.get_fqn(place_name)
                shape = "(("
                close = "))"
                prefix = "[INPUT]</br>" if place_spec.is_io_input else "[OUTPUT]</br>" if place_spec.is_io_output else ""
                lines.append(f'{indent}{place_fqn}{shape}"{prefix}{place_fqn}"{close}')

            for trans_name, trans_spec in spec.transitions.items():
                trans_fqn = spec.get_fqn(trans_name)

                # Check if this transition has BT or FSM integration
                integration = get_pn_integration(self._net_name, trans_name)

                if integration is not None:
                    # Add integration label to transition
                    if integration.integration_type == "bt":
                        # BT integration - embed the full BT tree diagram
                        self._add_bt_to_diagram(lines, integration, trans_fqn, indent)
                    elif integration.integration_type == "fsm":
                        # FSM integration - embed the full FSM state diagram
                        self._add_fsm_to_diagram(lines, integration, trans_fqn, indent)
                else:
                    # Vanilla transition
                    lines.append(f"{indent}{trans_fqn}[{trans_fqn}]")

            for arc in spec.arcs:
                source_fqn = arc.source.get_fqn()
                target_fqn = arc.target.get_fqn()

                # Check if source is a transition with FSM integration
                # If so, point from the subgraph instead of the transition node
                source_integration = None
                if hasattr(arc.source, 'local_name'):
                    source_integration = get_pn_integration(self._net_name, arc.source.local_name)

                # Check if target is a transition with FSM integration
                # If so, point to the subgraph instead of the transition node
                target_integration = None
                if hasattr(arc.target, 'local_name'):
                    target_integration = get_pn_integration(self._net_name, arc.target.local_name)

                # If source has FSM integration, use subgraph ID
                if source_integration and source_integration.integration_type == "fsm":
                    # Replace transition FQN with subgraph ID
                    # e.g., "SimpleFSMNet.process_with_fsm" -> "SimpleFSMNet.process_with_fsm_fsm"
                    source_fqn = f"{source_fqn}_fsm"

                # If source has BT integration, use subgraph ID
                if source_integration and source_integration.integration_type == "bt":
                    # Replace transition FQN with subgraph ID
                    # e.g., "JobQueueProcessor.route_by_priority" -> "JobQueueProcessor.route_by_priority_bt"
                    source_fqn = f"{source_fqn}_bt"

                # If target has FSM integration, use subgraph ID
                if target_integration and target_integration.integration_type == "fsm":
                    # Replace transition FQN with subgraph ID
                    # e.g., "SimpleFSMNet.process_with_fsm" -> "SimpleFSMNet.process_with_fsm_fsm"
                    target_fqn = f"{target_fqn}_fsm"

                # If target has BT integration, use subgraph ID
                if target_integration and target_integration.integration_type == "bt":
                    # Replace transition FQN with subgraph ID
                    # e.g., "JobQueueProcessor.route_by_priority" -> "JobQueueProcessor.route_by_priority_bt"
                    target_fqn = f"{target_fqn}_bt"

                if arc.weight > 1:
                    lines.append(f"{indent}{source_fqn} -->|weight={arc.weight}| {target_fqn}")
                else:
                    lines.append(f"{indent}{source_fqn} --> {target_fqn}")

        add_subnet(self._spec)

        return "\n".join(lines)

    def _add_fsm_to_diagram(self, lines: list, integration, trans_fqn: str, indent: str) -> None:
        """
        Add FSM state diagram as a subgraph embedded in the transition node.

        Generates a mini flowchart showing all FSM states and their transitions,
        embedded within the PN transition node using Mermaid's subgraph syntax.

        Args:
            lines: List of Mermaid diagram lines (modified in-place)
            integration: PNIntegration object with FSM metadata
            trans_fqn: Fully qualified name of the transition
            indent: Indentation string for diagram formatting
        """
        from ..septum.core import StateSpec, LabeledTransition, Push, Pop, Again, Retry, Unhandled, Repeat, Restart

        initial_state = integration.initial_state

        # Collect all reachable states from the initial state
        # Store both state objects and state names
        visited_states = {}  # state_name -> state_object
        to_visit = [initial_state]

        while to_visit:
            state = to_visit.pop()
            state_name = state.name if isinstance(state, StateSpec) else str(state)

            if state_name in visited_states:
                continue

            visited_states[state_name] = state

            # Get transitions from this state
            if isinstance(state, StateSpec) and hasattr(state, 'get_transitions'):
                transitions = state.get_transitions()

                # Handle single transition
                if not isinstance(transitions, list):
                    transitions = [transitions] if transitions else []

                for trans in transitions:
                    # Extract target states from transitions
                    if isinstance(trans, LabeledTransition):
                        # LabeledTransition has .transition attribute (the target)
                        target = trans.transition
                    elif isinstance(trans, (Again, Retry, Unhandled, Repeat, Restart)):
                        # Self-transitions, no new state to visit
                        continue
                    elif isinstance(trans, Pop):
                        # Pop doesn't add a specific state
                        continue
                    elif isinstance(trans, Push):
                        # Add all push states
                        for push_state in trans.push_states:
                            to_visit.append(push_state)
                        continue
                    elif isinstance(trans, StateSpec):
                        # Direct state reference
                        target = trans
                    else:
                        # Unknown transition type, skip
                        continue

                    # Add target state to visit list
                    if target is not None and target != state:
                        to_visit.append(target)

        # Create a subgraph for the FSM embedded in the transition
        subgraph_id = f"{trans_fqn}_fsm"
        lines.append(f'{indent}subgraph {subgraph_id}["{trans_fqn}</br>FSM: {integration.fsm_state_name}"]')

        # Add all FSM states as nodes within the subgraph
        # Use transition FQN as prefix to ensure unique IDs across all FSMs in the diagram
        # Sanitize trans_fqn to create valid Mermaid node IDs (replace dots with underscores)
        fsm_prefix = trans_fqn.replace(".", "_")
        state_ids = {}
        counter = 0
        for state_name in visited_states.keys():
            counter += 1
            # Create unique state ID using transition FQN prefix
            state_ids[state_name] = f"{fsm_prefix}_S{counter}"
            # Use simple state name (last component)
            simple_name = state_name.split(".")[-1]
            lines.append(f'{indent}    {state_ids[state_name]}["{simple_name}"]')

        # Mark initial state with unique start node
        initial_name = initial_state.name if isinstance(initial_state, StateSpec) else str(initial_state)
        start_node_id = f"{fsm_prefix}_start"
        lines.append(f'{indent}    {start_node_id}((start)) --> {state_ids[initial_name]}')

        # Add transitions between states
        for state_name, state_obj in visited_states.items():
            if not isinstance(state_obj, StateSpec):
                continue

            source_id = state_ids[state_name]

            # Get transitions
            transitions = state_obj.get_transitions()
            if not isinstance(transitions, list):
                transitions = [transitions] if transitions else []

            for trans in transitions:
                if isinstance(trans, LabeledTransition):
                    label = trans.label.name if hasattr(trans.label, 'name') else str(trans.label)
                    target = trans.transition

                    if isinstance(target, Again):
                        lines.append(f'{indent}    {source_id} -->|"again"| {source_id}')
                    elif isinstance(target, Retry):
                        lines.append(f'{indent}    {source_id} -->|"retry"| {source_id}')
                    elif isinstance(target, Unhandled):
                        lines.append(f'{indent}    {source_id} -->|"unhandled"| {source_id}')
                    elif isinstance(target, Repeat):
                        lines.append(f'{indent}    {source_id} -->|"repeat"| {source_id}')
                    elif isinstance(target, Restart):
                        lines.append(f'{indent}    {source_id} -->|"restart"| {source_id}')
                    elif isinstance(target, Pop):
                        pop_node_id = f"{fsm_prefix}_pop"
                        lines.append(f'{indent}    {source_id} -->|"{label}"| {pop_node_id}((pop))')
                    elif isinstance(target, Push):
                        # Push transitions to first pushed state
                        for push_state in target.push_states:
                            push_name = push_state.name if isinstance(push_state, StateSpec) else str(push_state)
                            if push_name in state_ids:
                                lines.append(f'{indent}    {source_id} -->|"{label} → push"| {state_ids[push_name]}')
                    elif isinstance(target, StateSpec):
                        target_name = target.name
                        if target_name in state_ids:
                            lines.append(f'{indent}    {source_id} -->|"{label}"| {state_ids[target_name]}')

                elif isinstance(trans, Again):
                    lines.append(f'{indent}    {source_id} -->|"again"| {source_id}')
                elif isinstance(trans, Retry):
                    lines.append(f'{indent}    {source_id} -->|"retry"| {source_id}')
                elif isinstance(trans, Unhandled):
                    lines.append(f'{indent}    {source_id} -->|"unhandled"| {source_id}')
                elif isinstance(trans, Repeat):
                    lines.append(f'{indent}    {source_id} -->|"repeat"| {source_id}')
                elif isinstance(trans, Restart):
                    lines.append(f'{indent}    {source_id} -->|"restart"| {source_id}')
                elif isinstance(trans, Pop):
                    pop_node_id = f"{fsm_prefix}_pop"
                    lines.append(f'{indent}    {source_id} -->|"pop"| {pop_node_id}((pop))')
                elif isinstance(trans, Push):
                    # Push transitions
                    for push_state in trans.push_states:
                        push_name = push_state.name if isinstance(push_state, StateSpec) else str(push_state)
                        if push_name in state_ids:
                            lines.append(f'{indent}    {source_id} -->|"push"| {state_ids[push_name]}')

        # Check if any pop transitions exist
        has_pop = False
        for state_obj in visited_states.values():
            if not isinstance(state_obj, StateSpec):
                continue
            state_transitions = state_obj.get_transitions()
            if not isinstance(state_transitions, list):
                state_transitions = [state_transitions] if state_transitions else []
            for trans in state_transitions:
                target = trans.transition if isinstance(trans, LabeledTransition) else trans
                if isinstance(target, Pop):
                    has_pop = True
                    break
            if has_pop:
                break

        if has_pop:
            pop_node_id = f"{fsm_prefix}_pop"
            lines.append(f'{indent}    {pop_node_id}((pop))')

        lines.append(f'{indent}end')

    def _add_bt_to_diagram(self, lines: list, integration, trans_fqn: str, indent: str) -> None:
        """
        Add BT tree diagram as a subgraph embedded in the transition node.

        Generates a mini flowchart showing all BT nodes and their structure,
        embedded within the PN transition node using Mermaid's subgraph syntax.

        Args:
            lines: List of Mermaid diagram lines (modified in-place)
            integration: PNIntegration object with BT metadata
            trans_fqn: Fully qualified name of the transition
            indent: Indentation string for diagram formatting
        """
        from ..rhizomorph.core import NodeSpecKind, _bt_expand_children

        bt_tree = integration.bt_tree

        # Get BT tree name
        if hasattr(bt_tree, '_tree_name'):
            bt_name = bt_tree._tree_name
        elif hasattr(bt_tree, '_spec') and hasattr(bt_tree._spec, 'name'):
            bt_name = bt_tree._spec.name
        elif hasattr(bt_tree, '__name__'):
            bt_name = bt_tree.__name__
        else:
            bt_name = "BT"

        # Get the root node spec
        if hasattr(bt_tree, 'root'):
            root_spec = bt_tree.root
        elif hasattr(bt_tree, '_spec'):
            root_spec = bt_tree._spec
        else:
            # Fallback: just show the name without details
            label = f"{trans_fqn}</br>BT: {bt_name}"
            lines.append(f'{indent}{trans_fqn}["{label}"]')
            return

        # Create a subgraph for the BT embedded in the transition
        subgraph_id = f"{trans_fqn}_bt"
        lines.append(f'{indent}subgraph {subgraph_id}["{trans_fqn}</br>BT: {bt_name}"]')

        # Helper to ensure children are expanded (similar to _generate_mermaid)
        def ensure_children(spec):
            """Expand children of composite nodes using the factory function."""
            if hasattr(spec, 'children') and spec.children:
                # Already expanded
                return spec.children

            match spec.kind:
                case NodeSpecKind.SEQUENCE | NodeSpecKind.SELECTOR | NodeSpecKind.PARALLEL:
                    factory = spec.payload.get("factory")
                    if factory:
                        spec.children = _bt_expand_children(factory)
                    return spec.children if hasattr(spec, 'children') else []
                case NodeSpecKind.SUBTREE:
                    subtree_root = spec.payload.get("root")
                    if subtree_root:
                        spec.children = [subtree_root]
                    return spec.children if hasattr(spec, 'children') else []
                case _:
                    return []

        # Helper to add BT nodes
        # Use transition FQN as prefix to ensure unique IDs across all BTs in the diagram
        # Sanitize trans_fqn to create valid Mermaid node IDs (replace dots with underscores)
        bt_prefix = trans_fqn.replace(".", "_")
        node_counter = 0
        node_map = {}

        def add_bt_nodes(node_spec, parent_id=None, edge_label=""):
            nonlocal node_counter
            node_counter += 1
            # Create unique node ID using transition FQN prefix
            node_id = f"{bt_prefix}_BT{node_counter}"
            node_map[id(node_spec)] = node_id

            # Get node kind and name
            kind = node_spec.kind if hasattr(node_spec, 'kind') else None
            name = node_spec.name if hasattr(node_spec, 'name') else "node"

            # Format node based on kind
            if kind == NodeSpecKind.ACTION:
                shape = "((ACTION<br/>"
                close = "))"
            elif kind == NodeSpecKind.CONDITION:
                shape = "{{CONDITION<br/>"
                close = "}}"
            elif kind == NodeSpecKind.SEQUENCE:
                shape = "[SEQUENCE<br/>"
                close = "]"
            elif kind == NodeSpecKind.SELECTOR:
                shape = "[SELECTOR<br/>"
                close = "]"
            elif kind == NodeSpecKind.PARALLEL:
                shape = "[PARALLEL<br/>"
                close = "]"
            else:
                shape = "["
                close = "]"

            lines.append(f'{indent}    {node_id}{shape}{name}{close}')

            # Add edge from parent if this is not the root
            if parent_id:
                if edge_label:
                    lines.append(f'{indent}    {parent_id} -->|{edge_label}| {node_id}')
                else:
                    lines.append(f'{indent}    {parent_id} --> {node_id}')

            # Expand and recursively add children
            children = ensure_children(node_spec)
            for child in children:
                add_bt_nodes(child, node_id)

        # Start adding nodes from the root
        if root_spec:
            add_bt_nodes(root_spec)

        lines.append(f'{indent}end')


def create_pn_net_decorator():
    """
    Create the @pn_net decorator that supports BT-in-PN and FSM-in-PN integration.

    This decorator wraps Hypha's @pn.net to add BT/FSM integration support.
    """
    from ..hypha.core.builder import pn as hypha_pn

    def pn_net(func: Callable) -> Callable:
        """
        Decorator to define a Petri net with BT-in-PN and FSM-in-PN support.

        Usage:
            # Vanilla transition (no integration)
            @pn_net
            def MyNet(builder):
                input_q = builder.place("input_q", type=PlaceType.QUEUE)
                output_q = builder.place("output_q", type=PlaceType.QUEUE)

                @builder.transition()
                async def process(consumed, bb, timebase):
                    yield {output_q: token}

                builder.arc(input_q, process)
                builder.arc(process, output_q)

            # With BT integration
            @pn_net
            def MyNet(builder):
                input_q = builder.place("input_q", type=PlaceType.QUEUE)
                output_q = builder.place("output_q", type=PlaceType.QUEUE)

                @builder.transition(bt=MyBT, bt_mode="token", outputs=[output_q])
                async def route(consumed, bb, timebase):
                    pass  # BT handles routing

                builder.arc(input_q, route)
                builder.arc(route, output_q)

            # With FSM integration
            @pn_net
            def MyNet(builder):
                input_q = builder.place("input_q", type=PlaceType.QUEUE)
                success_q = builder.place("success_q", type=PlaceType.QUEUE)
                failure_q = builder.place("failure_q", type=PlaceType.QUEUE)

                @builder.transition(fsm=MyFSMState, outputs=[success_q, failure_q])
                async def process(consumed, bb, timebase):
                    pass  # FSM handles processing

                builder.arc(input_q, process)
                builder.arc(process, success_q)
                builder.arc(process, failure_q)
        """
        net_name = func.__name__

        # Create wrapped builder and get spec
        net_spec, wrapped_builder = create_net_wrapper(func, net_name)

        # Create a net function that has the spec attached
        # (like Hypha's @pn.net does)
        def net_func():
            return net_spec

        # Attach spec to the function (like Hypha does)
        net_func._spec = net_spec  # type: ignore[attr-defined]
        net_func.__name__ = net_name
        net_func.__qualname__ = func.__qualname__
        net_func.__module__ = func.__module__
        net_func.__doc__ = func.__doc__

        return net_func

    return pn_net


# Create the singleton pn_net decorator
pn_net = create_pn_net_decorator()


# Create a pn namespace for backwards compatibility
class _PN:
    """Namespace for PN decorators to match hypha.core.pn API."""

    net = staticmethod(pn_net)


# Create singleton instance
pn = _PN()
