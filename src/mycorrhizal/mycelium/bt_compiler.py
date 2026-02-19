#!/usr/bin/env python3
"""
BT-to-PN Compiler for Mycelium

This module provides the Behavior Tree to Petri Net compiler as part of the
Mycelium unified orchestration layer. The compiler transforms Rhizomorph behavior
trees into Hypha Petri nets for execution.

Key features:
- Compile BT trees to Petri net specifications
- Configurable timing parameters (running_delay, retry_delay)
- Peephole optimization for reduced net size
- Support for all BT node types (actions, conditions, composites, decorators)

Example:
    >>> from mycorrhizal.mycelium import compile_bt_to_pn
    >>> from mycorrhizal.rhizomorph.core import bt
    >>>
    >>> @bt.tree
    >>> def MyBT():
    >>>     @bt.action
    >>>     async def my_action(bb):
    >>>         return Status.SUCCESS
    >>>
    >>>     @bt.root
    >>>     @bt.sequence
    >>>     def root():
    >>>         yield my_action
    >>>
    >>> # Compile to Petri net
    >>> pn_spec = compile_bt_to_pn(MyBT)
    >>>
    >>> # Use with Mycelium's PN runner
    >>> from mycorrhizal.mycelium import PNRunner
    >>> runner = PNRunner(pn_spec, blackboard=bb)
"""

from __future__ import annotations

__all__ = [
    "compile_bt_to_pn",
    "BTtoPNCompiler",
    "StatusToken",
    "MatchToken",
    "ParallelCounterToken",
]

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

from ..hypha.core.specs import NetSpec
from ..hypha.core.builder import NetBuilder
from ..rhizomorph.core import Status

if TYPE_CHECKING:
    from ..rhizomorph.core import NodeSpec


# =============================================================================
# Token Types - Status as Colored Tokens
# =============================================================================


@dataclass(frozen=True)
class StatusToken:
    """Base token representing BT status flow.

    Attributes:
        status: The BT status (SUCCESS, FAILURE, RUNNING)
        data: Optional payload data
        attempts_remaining: For retry tracking in decorators
        last_start_time: For RateLimit tracking (when child started)
        next_allowed_time: For RateLimit tracking (when next start is allowed)
        child_running: For DoWhile tracking
        iteration_count: For DoWhile iteration tracking
    """
    status: Status
    data: Any = None
    attempts_remaining: int = 0
    last_start_time: float = 0.0
    next_allowed_time: float = 0.0
    child_running: bool = False
    iteration_count: int = 0


@dataclass(frozen=True)
class MatchToken:
    """Token carrying match-specific state for Match decorator.

    The Match decorator needs to remember which case was matched across
    RUNNING ticks. This token wraps a StatusToken with match metadata.

    Attributes:
        base: The underlying status token
        matched_case_idx: Index of matched case (None = not yet matched)
        key_value: Cached value from key_fn evaluation
    """
    base: StatusToken
    matched_case_idx: int | None = None
    key_value: Any = None


@dataclass(frozen=True)
class ParallelCounterToken:
    """Token carrying parallel completion count state.

    The Parallel node needs to count successes and failures to determine
    when success_threshold is reached. This token carries the running counts.

    Attributes:
        success_count: Number of children that have succeeded
        failure_count: Number of children that have failed
        total_children: Total number of children (N)
        success_threshold: Minimum successes needed (threshold)
    """
    success_count: int
    failure_count: int
    total_children: int
    success_threshold: int


# =============================================================================
# Compilation Context
# =============================================================================


@dataclass
class CompilationContext:
    """Holds state during compilation.

    Attributes:
        builder: The NetBuilder being used
        blackboard_type: Optional type hint for blackboard
        _place_counter: Internal counter for unique place names
        _transition_counter: Internal counter for unique transition names
        running_delay: Delay between RUNNING retry attempts (seconds)
        retry_delay: Delay between Retry decorator attempts (seconds)
    """
    builder: NetBuilder
    blackboard_type: Optional[type] = None
    _place_counter: int = field(default=0, repr=False)
    _transition_counter: int = field(default=0, repr=False)
    running_delay: float = 0.1
    retry_delay: float = 0.0

    def fresh_place_name(self, prefix: str = "p") -> str:
        """Generate a unique place name."""
        self._place_counter += 1
        return f"{prefix}_{self._place_counter}"

    def fresh_transition_name(self, prefix: str = "t") -> str:
        """Generate a unique transition name."""
        self._transition_counter += 1
        return f"{prefix}_{self._transition_counter}"


@dataclass
class CompiledNode:
    """Result of compiling a BT node to PN structures.

    Attributes:
        entry: The entry place (receives tick tokens)
        success_exit: Emits on SUCCESS
        failure_exit: Emits on FAILURE
        running_exit: Emits on RUNNING (optional)
        places: All places created by this node
        transitions: All transitions created by this node
    """
    entry: Any  # PlaceRef
    success_exit: Any  # PlaceRef
    failure_exit: Any  # PlaceRef
    running_exit: Optional[Any] = None  # PlaceRef
    places: List[Any] = field(default_factory=list)
    transitions: List[Any] = field(default_factory=list)


# =============================================================================
# Compiler
# =============================================================================


class BTtoPNCompiler:
    """
    Compiles Rhizomorph behavior trees to Hypha Petri nets.

    The compilation strategy transforms each BT node into a subnet with:
    - entry place (receives tick tokens)
    - success_exit place (emits on SUCCESS)
    - failure_exit place (emits on FAILURE)
    - running_exit place (emits on RUNNING, optional)

    Actions/Conditions become transitions with timed self-loops for RUNNING.
    Composites wire their children's exits to entries based on their semantics.

    Configuration Parameters:
        name: Name for the compiled Petri net
        running_delay: Delay between RUNNING retry attempts (default 0.1s)
        retry_delay: Delay between Retry decorator attempts (default 0.0s)
    """

    def __init__(self, name: str = "CompiledBT", running_delay: float = 0.1, retry_delay: float = 0.0):
        """Initialize the compiler.

        Args:
            name: Name for the compiled Petri net
            running_delay: Delay between RUNNING retry attempts in seconds
            retry_delay: Delay between Retry decorator attempts in seconds
        """
        self.name = name
        self.running_delay = running_delay
        self.retry_delay = retry_delay

    def compile(self, tree_spec: "NodeSpec") -> NetSpec:
        """
        Compile a behavior tree specification to a Petri net specification.

        Args:
            tree_spec: The root NodeSpec of the behavior tree

        Returns:
            NetSpec that can be executed by Hypha
        """
        builder = NetBuilder(self.name)
        ctx = CompilationContext(
            builder=builder,
            running_delay=self.running_delay,
            retry_delay=self.retry_delay
        )

        # Create the main entry point
        entry = builder.place("entry")

        # Create result places (external interface)
        success_result = builder.place("success")
        failure_result = builder.place("failure")
        running_result = builder.place("running")

        # Compile the tree
        compiled = self._compile_node(tree_spec, ctx, entry)

        # Wire the compiled exits to the result places
        builder.forward(compiled.success_exit, success_result, name="emit_success")
        builder.forward(compiled.failure_exit, failure_result, name="emit_failure")
        if compiled.running_exit is not None:
            builder.forward(compiled.running_exit, running_result, name="emit_running")

        # Peephole optimization: eliminate pure forward transitions
        pre_opt_count = len(builder.spec.transitions)
        self._optimize_net(builder.spec)
        post_opt_count = len(builder.spec.transitions)
        eliminated = pre_opt_count - post_opt_count
        if eliminated > 0:
            print(f"  [OPTIMIZATION] Eliminated {eliminated} pure forward transitions ({pre_opt_count} -> {post_opt_count})")

        return builder.spec

    def _optimize_net(self, spec: NetSpec) -> None:
        """
        Eliminate pure forward transitions (peephole optimization).

        A "pure forward" transition:
        - Has exactly one input place
        - Has exactly one output place
        - Passes the token through unchanged (identity transformation)
        - delay=0 (no timing constraints)
        - NO guard (guards can prevent firing, so we can't optimize them away)
        - Output place has consumers (not an external output)

        Optimization runs in a loop until no more optimizations are possible,
        to handle chains of forward transitions.
        """
        from ..hypha.core.specs import ArcSpec

        total_eliminated = 0
        max_iterations = 100  # Safety limit

        for iteration in range(max_iterations):
            # Rebuild indices each iteration (structure may have changed)
            place_to_trans_consumers = {}
            trans_to_input_places = {}
            trans_to_output_places = {}
            place_from_trans_producers = {}

            for arc in spec.arcs:
                if isinstance(arc.source, type(spec.places[list(spec.places.keys())[0]])):
                    # PlaceRef -> Transition
                    trans_name = arc.target.local_name
                    place_to_trans_consumers.setdefault(arc.source, []).append((arc.target, arc))
                    trans_to_input_places.setdefault(trans_name, []).append((arc.source, arc, arc.target))
                elif isinstance(arc.target, type(spec.places[list(spec.places.keys())[0]])):
                    # Transition -> PlaceRef
                    trans_name = arc.source.local_name
                    trans_to_output_places.setdefault(trans_name, []).append((arc.target, arc, arc.source))
                    place_from_trans_producers.setdefault(arc.target, []).append((arc.source, arc))

            # Find ONE pure forward transition to eliminate (eliminate one at a time)
            pure_forward = None
            for trans_name, trans_spec in spec.transitions.items():
                inputs = trans_to_input_places.get(trans_name, [])
                outputs = trans_to_output_places.get(trans_name, [])

                if len(inputs) == 1 and len(outputs) == 1:
                    if trans_spec.delay is None or trans_spec.delay == 0:
                        # CRITICAL: Only optimize if there's NO guard
                        # Guards can prevent transitions from firing, so we can't eliminate them
                        if trans_spec.guard is not None:
                            continue

                        input_place, _, trans_ref = inputs[0]
                        output_place, _, _ = outputs[0]

                        # Only eliminate if output place has consumers (not external output)
                        consumers = place_to_trans_consumers.get(output_place, [])
                        if consumers:
                            # Also check: output place should only have ONE producer
                            # (otherwise we'd create duplicate tokens)
                            producers = place_from_trans_producers.get(output_place, [])
                            if len(producers) == 1:
                                pure_forward = (trans_ref, trans_name, input_place, output_place, consumers)
                                break

            if not pure_forward:
                break  # No more optimizations possible

            trans_ref, trans_name, input_place, output_place, consumers = pure_forward

            # Rewire consumers of output_place to consume from input_place
            arcs_to_remove = []
            arcs_to_add = []

            for consumer_trans, old_arc in consumers:
                arcs_to_remove.append(old_arc)
                arcs_to_add.append(ArcSpec(input_place, consumer_trans))

            # Remove the forward transition's arcs
            for _, arc, _ in trans_to_input_places.get(trans_name, []):
                arcs_to_remove.append(arc)
            for _, arc, _ in trans_to_output_places.get(trans_name, []):
                arcs_to_remove.append(arc)

            # Apply changes
            spec.arcs = [a for a in spec.arcs if a not in arcs_to_remove]
            spec.arcs.extend(arcs_to_add)

            # Remove the transition
            spec.transitions.pop(trans_name, None)

            # Remove the output place (it had only one producer which we removed)
            spec.places.pop(output_place.local_name, None)

            total_eliminated += 1

        if total_eliminated > 0:
            print(f"      [OPT] Eliminated {total_eliminated} pure forward transitions")

    def _compile_node(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """
        Compile a single node spec to PN structures.

        Dispatches to the appropriate handler based on node kind.
        """
        from ..rhizomorph.core import NodeSpecKind

        kind = node_spec.kind

        if kind == NodeSpecKind.ACTION:
            return self._compile_action(node_spec, ctx, entry)
        elif kind == NodeSpecKind.CONDITION:
            return self._compile_condition(node_spec, ctx, entry)
        elif kind == NodeSpecKind.SEQUENCE:
            return self._compile_sequence(node_spec, ctx, entry)
        elif kind == NodeSpecKind.SELECTOR:
            return self._compile_selector(node_spec, ctx, entry)
        elif kind == NodeSpecKind.PARALLEL:
            return self._compile_parallel(node_spec, ctx, entry)
        elif kind == NodeSpecKind.DECORATOR:
            return self._compile_decorator(node_spec, ctx, entry)
        elif kind == NodeSpecKind.DO_WHILE:
            return self._compile_do_while(node_spec, ctx, entry)
        elif kind == NodeSpecKind.TRY_CATCH:
            return self._compile_try_catch(node_spec, ctx, entry)
        elif kind == NodeSpecKind.MATCH:
            return self._compile_match(node_spec, ctx, entry)
        elif kind == NodeSpecKind.SUBTREE:
            return self._compile_subtree(node_spec, ctx, entry)
        else:
            raise NotImplementedError(f"Cannot compile node kind: {kind}")

    def _compile_action(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile an action node to a transition with timed self-loop for RUNNING."""
        builder = ctx.builder
        func = node_spec.payload

        # Create action_entry place
        action_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_entry"))
        builder.forward(entry, action_entry, name=f"{node_spec.name}_from_parent")

        # Create exit places
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))

        # Internal running place
        running_internal = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running_internal"))

        # Signal place (for parent to observe RUNNING)
        running_signal = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running_signal"))

        # Create the action transition
        trans_name = ctx.fresh_transition_name(f"{node_spec.name}_action")

        def make_action_handler(action_func, success_place, failure_place, running_place):
            async def handler(consumed, bb, timebase):
                import asyncio
                result = action_func(bb)
                if asyncio.iscoroutine(result):
                    result = await result

                # Coerce to Status
                if isinstance(result, Status):
                    status = result
                elif isinstance(result, bool):
                    status = Status.SUCCESS if result else Status.FAILURE
                elif result is None:
                    status = Status.SUCCESS
                else:
                    status = Status.SUCCESS

                # Preserve attempts_remaining
                attempts_remaining = 0
                if consumed and isinstance(consumed[0], StatusToken):
                    attempts_remaining = consumed[0].attempts_remaining

                # Route to appropriate exit
                if status == Status.SUCCESS:
                    yield {success_place: StatusToken(status=status, attempts_remaining=attempts_remaining)}
                elif status == Status.RUNNING:
                    yield {running_place: StatusToken(status=status, attempts_remaining=attempts_remaining)}
                else:
                    yield {failure_place: StatusToken(status=status, attempts_remaining=attempts_remaining)}
            handler.__name__ = trans_name
            return handler

        action_handler = make_action_handler(func, success_exit, failure_exit, running_internal)
        trans_ref = builder.transition(delay=0.0)(action_handler)

        builder.arc(action_entry, trans_ref)
        builder.arc(trans_ref, success_exit)
        builder.arc(trans_ref, failure_exit)
        builder.arc(trans_ref, running_internal)

        # Create the retry transition for RUNNING
        retry_name = ctx.fresh_transition_name(f"{node_spec.name}_retry")

        def make_retry_handler(entry_place, signal_place):
            async def handler(consumed, bb, timebase):
                yield {
                    entry_place: StatusToken(status=Status.SUCCESS),
                    signal_place: StatusToken(status=Status.RUNNING),
                }
            handler.__name__ = retry_name
            return handler

        retry_handler = make_retry_handler(action_entry, running_signal)
        retry_ref = builder.transition(delay=ctx.running_delay)(retry_handler)

        builder.arc(running_internal, retry_ref)
        builder.arc(retry_ref, action_entry)
        builder.arc(retry_ref, running_signal)

        all_transitions = [trans_ref, retry_ref]

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_signal,
            places=[entry, action_entry, success_exit, failure_exit, running_internal, running_signal],
            transitions=all_transitions
        )

    def _compile_condition(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a condition node to a transition with timed self-loop for RUNNING."""
        builder = ctx.builder
        func = node_spec.payload

        # Create condition_entry place
        condition_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_entry"))
        builder.forward(entry, condition_entry, name=f"{node_spec.name}_from_parent")

        # Create exit places
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))

        # Internal running place
        running_internal = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running_internal"))

        # Signal place
        running_signal = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running_signal"))

        trans_name = ctx.fresh_transition_name(f"{node_spec.name}_condition")

        def make_condition_handler(cond_func, success_place, failure_place, running_place):
            async def handler(consumed, bb, timebase):
                import asyncio
                result = cond_func(bb)
                if asyncio.iscoroutine(result):
                    result = await result

                # Preserve attempts_remaining
                attempts_remaining = 0
                if consumed and isinstance(consumed[0], StatusToken):
                    attempts_remaining = consumed[0].attempts_remaining

                # Coerce to status
                if isinstance(result, Status):
                    status = result
                    if status == Status.SUCCESS:
                        yield {success_place: StatusToken(status=Status.SUCCESS, attempts_remaining=attempts_remaining)}
                    elif status == Status.RUNNING:
                        yield {running_place: StatusToken(status=Status.RUNNING, attempts_remaining=attempts_remaining)}
                    else:
                        yield {failure_place: StatusToken(status=Status.FAILURE, attempts_remaining=attempts_remaining)}
                else:
                    # Boolean result
                    passed = bool(result)
                    if passed:
                        yield {success_place: StatusToken(status=Status.SUCCESS, attempts_remaining=attempts_remaining)}
                    else:
                        yield {failure_place: StatusToken(status=Status.FAILURE, attempts_remaining=attempts_remaining)}
            handler.__name__ = trans_name
            return handler

        condition_handler = make_condition_handler(func, success_exit, failure_exit, running_internal)
        trans_ref = builder.transition(delay=0.0)(condition_handler)

        builder.arc(condition_entry, trans_ref)
        builder.arc(trans_ref, success_exit)
        builder.arc(trans_ref, failure_exit)
        builder.arc(trans_ref, running_internal)

        # Create the retry transition for RUNNING
        retry_name = ctx.fresh_transition_name(f"{node_spec.name}_retry")

        def make_retry_handler(entry_place, signal_place):
            async def handler(consumed, bb, timebase):
                yield {
                    entry_place: StatusToken(status=Status.SUCCESS),
                    signal_place: StatusToken(status=Status.RUNNING),
                }
            handler.__name__ = retry_name
            return handler

        retry_handler = make_retry_handler(condition_entry, running_signal)
        retry_ref = builder.transition(delay=ctx.running_delay)(retry_handler)

        builder.arc(running_internal, retry_ref)
        builder.arc(retry_ref, condition_entry)
        builder.arc(retry_ref, running_signal)

        all_transitions = [trans_ref, retry_ref]

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_signal,
            places=[entry, condition_entry, success_exit, failure_exit, running_internal, running_signal],
            transitions=all_transitions
        )

    def _compile_sequence(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a sequence to a chain of subnets."""
        from ..rhizomorph.core import _bt_expand_children

        builder = ctx.builder

        # Get children
        factory = node_spec.payload.get("factory")
        children = _bt_expand_children(factory)

        # Create shared exits (used for all non-last child exits)
        seq_failure = builder.place(ctx.fresh_place_name("seq_failure"))
        seq_running = builder.place(ctx.fresh_place_name("seq_running"))

        if not children:
            # Empty sequence: create a success exit and forward to it
            success_exit = builder.place(ctx.fresh_place_name("seq_success"))
            builder.forward(entry, success_exit, name="empty_seq")
            return CompiledNode(
                entry=entry,
                success_exit=success_exit,
                failure_exit=seq_failure,
                running_exit=seq_running,
                places=[entry, success_exit, seq_failure, seq_running],
                transitions=[]
            )

        # Compile children in sequence
        current_entry = entry
        all_places = [entry, seq_failure, seq_running]
        all_transitions = []

        for i, child_spec in enumerate(children):
            is_last = (i == len(children) - 1)

            compiled_child = self._compile_node(child_spec, ctx, current_entry)
            all_places.extend(compiled_child.places)
            all_transitions.extend(compiled_child.transitions)

            if is_last:
                # Last child: forward failure to seq_failure, running to seq_running
                # Return child's success_exit as our success_exit
                builder.forward(
                    compiled_child.failure_exit,
                    seq_failure,
                    name=ctx.fresh_transition_name("seq_last_fail")
                )
                if compiled_child.running_exit is not None:
                    builder.forward(
                        compiled_child.running_exit,
                        seq_running,
                        name=ctx.fresh_transition_name("seq_last_running")
                    )
                return CompiledNode(
                    entry=entry,
                    success_exit=compiled_child.success_exit,
                    failure_exit=seq_failure,
                    running_exit=seq_running,
                    places=all_places,
                    transitions=all_transitions
                )
            else:
                # Non-last child: create intermediate place, wire up exits
                current_entry = builder.place(ctx.fresh_place_name("seq_intermediate"))
                forward_name = ctx.fresh_transition_name("seq_continue")
                builder.forward(
                    compiled_child.success_exit,
                    current_entry,
                    name=forward_name
                )
                builder.forward(
                    compiled_child.failure_exit,
                    seq_failure,
                    name=ctx.fresh_transition_name("seq_child_fail")
                )
                if compiled_child.running_exit is not None:
                    builder.forward(
                        compiled_child.running_exit,
                        seq_running,
                        name=ctx.fresh_transition_name("seq_child_running")
                    )
                all_places.append(current_entry)

        raise RuntimeError("Unexpected end of sequence compilation")

    def _compile_selector(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a selector to parallel branches."""
        from ..rhizomorph.core import _bt_expand_children

        builder = ctx.builder

        # Get children
        factory = node_spec.payload.get("factory")
        children = _bt_expand_children(factory)

        # Create final exits
        final_success = builder.place(ctx.fresh_place_name("sel_success"))
        final_failure = builder.place(ctx.fresh_place_name("sel_failure"))
        final_running = builder.place(ctx.fresh_place_name("sel_running"))

        if not children:
            builder.forward(entry, final_failure, name="empty_sel")
            return CompiledNode(
                entry=entry,
                success_exit=final_success,
                failure_exit=final_failure,
                running_exit=final_running,
                places=[entry, final_success, final_failure, final_running],
                transitions=[]
            )

        all_places = [entry, final_success, final_failure, final_running]
        all_transitions = []

        current_entry = entry

        for i, child_spec in enumerate(children):
            is_last = (i == len(children) - 1)

            compiled_child = self._compile_node(child_spec, ctx, current_entry)
            all_places.extend(compiled_child.places)
            all_transitions.extend(compiled_child.transitions)

            builder.forward(
                compiled_child.success_exit,
                final_success,
                name=ctx.fresh_transition_name("sel_success")
            )

            if not is_last:
                current_entry = builder.place(ctx.fresh_place_name("sel_next"))
                builder.forward(
                    compiled_child.failure_exit,
                    current_entry,
                    name=ctx.fresh_transition_name("sel_try_next")
                )
                if compiled_child.running_exit is not None:
                    builder.forward(
                        compiled_child.running_exit,
                        final_running,
                        name=ctx.fresh_transition_name("sel_running")
                    )
                all_places.append(current_entry)
            else:
                builder.forward(
                    compiled_child.failure_exit,
                    final_failure,
                    name=ctx.fresh_transition_name("sel_all_failed")
                )
                if compiled_child.running_exit is not None:
                    builder.forward(
                        compiled_child.running_exit,
                        final_running,
                        name=ctx.fresh_transition_name("sel_last_running")
                    )

        return CompiledNode(
            entry=entry,
            success_exit=final_success,
            failure_exit=final_failure,
            running_exit=final_running,
            places=all_places,
            transitions=all_transitions
        )

    def _compile_parallel(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a parallel node with threshold-based success counting.

        Uses a dual-transition pattern to handle token accumulation:
        - First completion uses _init transition (no counter input)
        - Subsequent completions use _update transition (consumes and produces counter)

        This avoids race conditions by ensuring counter updates are serialized
        through the counter place as a required input.
        """
        from ..rhizomorph.core import _bt_expand_children

        builder = ctx.builder

        # Get parameters
        payload = node_spec.payload
        success_threshold = payload.get("success_threshold", 1)

        # Get children
        factory = payload.get("factory")
        children = _bt_expand_children(factory)

        # Create final exits
        final_success = builder.place(ctx.fresh_place_name("par_success"))
        final_failure = builder.place(ctx.fresh_place_name("par_failure"))
        final_running = builder.place(ctx.fresh_place_name("par_running"))

        if not children:
            builder.forward(entry, final_success, name="empty_par")
            return CompiledNode(
                entry=entry,
                success_exit=final_success,
                failure_exit=final_failure,
                running_exit=final_running,
                places=[entry, final_success, final_failure, final_running],
                transitions=[]
            )

        # Create counter place to track completions
        # Initially empty, first completion initializes it
        counter_place = builder.place(ctx.fresh_place_name("par_counter"))

        # Calculate thresholds
        total_children = len(children)
        failure_threshold = total_children - success_threshold + 1

        all_places = [entry, final_success, final_failure, final_running, counter_place]
        all_transitions = []
        child_entries = []

        for child_spec in children:
            child_entry = builder.place(ctx.fresh_place_name("par_child_in"))
            child_entries.append(child_entry)
            all_places.append(child_entry)

            compiled_child = self._compile_node(child_spec, ctx, child_entry)
            all_places.extend(compiled_child.places)
            all_transitions.extend(compiled_child.transitions)

            # ========== Success handling ==========

            # First success transition (no counter input)
            def make_success_init_handler(counter, success_place, threshold, total):
                async def handler(consumed, bb, timebase):
                    # First success: check if threshold is already met
                    if 1 >= threshold:
                        yield {success_place: StatusToken(status=Status.SUCCESS)}
                    else:
                        # Initialize counter with first success
                        yield {
                            counter: ParallelCounterToken(
                                success_count=1,
                                failure_count=0,
                                total_children=total,
                                success_threshold=threshold
                            )
                        }
                return handler

            success_init_name = ctx.fresh_transition_name("par_success_init")
            success_init_handler = make_success_init_handler(counter_place, final_success, success_threshold, total_children)
            success_init_handler.__name__ = success_init_name
            success_init_trans = builder.transition()(success_init_handler)
            all_transitions.append(success_init_trans)

            builder.arc(compiled_child.success_exit, success_init_trans)
            builder.arc(success_init_trans, final_success)
            builder.arc(success_init_trans, counter_place)

            # Subsequent success transition (consumes counter, produces updated counter)
            def make_success_update_handler(counter, success_place, threshold, total):
                async def handler(consumed, bb, timebase):
                    # consumed[0] is the counter token
                    token = consumed[0]
                    if not isinstance(token, ParallelCounterToken):
                        return

                    new_count = token.success_count + 1
                    # Check if we've reached the success threshold
                    if new_count >= threshold:
                        yield {success_place: StatusToken(status=Status.SUCCESS)}
                    else:
                        # Update counter and continue waiting
                        yield {
                            counter: ParallelCounterToken(
                                success_count=new_count,
                                failure_count=token.failure_count,
                                total_children=total,
                                success_threshold=threshold
                            )
                        }
                return handler

            success_update_name = ctx.fresh_transition_name("par_success_update")
            success_update_handler = make_success_update_handler(counter_place, final_success, success_threshold, total_children)
            success_update_handler.__name__ = success_update_name
            success_update_trans = builder.transition()(success_update_handler)
            all_transitions.append(success_update_trans)

            builder.arc(compiled_child.success_exit, success_update_trans)
            builder.arc(counter_place, success_update_trans)  # Consume counter
            builder.arc(success_update_trans, final_success)
            builder.arc(success_update_trans, counter_place)  # Produce updated counter

            # ========== Failure handling ==========

            # First failure transition (no counter input)
            def make_failure_init_handler(counter, failure_place, fail_threshold, total, success_thresh):
                async def handler(consumed, bb, timebase):
                    # First failure: check if failure threshold is already met
                    if 1 >= fail_threshold:
                        yield {failure_place: StatusToken(status=Status.FAILURE)}
                    else:
                        # Initialize counter with first failure
                        yield {
                            counter: ParallelCounterToken(
                                success_count=0,
                                failure_count=1,
                                total_children=total,
                                success_threshold=success_thresh
                            )
                        }
                return handler

            failure_init_name = ctx.fresh_transition_name("par_failure_init")
            failure_init_handler = make_failure_init_handler(counter_place, final_failure, failure_threshold, total_children, success_threshold)
            failure_init_handler.__name__ = failure_init_name
            failure_init_trans = builder.transition()(failure_init_handler)
            all_transitions.append(failure_init_trans)

            builder.arc(compiled_child.failure_exit, failure_init_trans)
            builder.arc(failure_init_trans, final_failure)
            builder.arc(failure_init_trans, counter_place)

            # Subsequent failure transition (consumes counter, produces updated counter)
            def make_failure_update_handler(counter, failure_place, fail_threshold, total, success_thresh):
                async def handler(consumed, bb, timebase):
                    # consumed[0] is the counter token
                    token = consumed[0]
                    if not isinstance(token, ParallelCounterToken):
                        return

                    new_count = token.failure_count + 1
                    # Check if we've exceeded the failure threshold
                    if new_count >= fail_threshold:
                        yield {failure_place: StatusToken(status=Status.FAILURE)}
                    else:
                        # Update counter and continue waiting
                        yield {
                            counter: ParallelCounterToken(
                                success_count=token.success_count,
                                failure_count=new_count,
                                total_children=total,
                                success_threshold=success_thresh
                            )
                        }
                return handler

            failure_update_name = ctx.fresh_transition_name("par_failure_update")
            failure_update_handler = make_failure_update_handler(counter_place, final_failure, failure_threshold, total_children, success_threshold)
            failure_update_handler.__name__ = failure_update_name
            failure_update_trans = builder.transition()(failure_update_handler)
            all_transitions.append(failure_update_trans)

            builder.arc(compiled_child.failure_exit, failure_update_trans)
            builder.arc(counter_place, failure_update_trans)  # Consume counter
            builder.arc(failure_update_trans, final_failure)
            builder.arc(failure_update_trans, counter_place)  # Produce updated counter

            # Child running -> propagate directly (no counting for RUNNING)
            if compiled_child.running_exit is not None:
                builder.forward(
                    compiled_child.running_exit,
                    final_running,
                    name=ctx.fresh_transition_name("par_r_join")
                )

        # Fork transition
        def make_fork(child_entries):
            async def fork_handler(consumed, bb, timebase):
                result = {}
                for child_entry in child_entries:
                    result[child_entry] = StatusToken(status=Status.SUCCESS)
                yield result
            return fork_handler

        fork_name = ctx.fresh_transition_name("par_fork")
        fork_handler = make_fork(child_entries)
        fork_handler.__name__ = fork_name
        fork_trans = builder.transition()(fork_handler)
        all_transitions.append(fork_trans)

        builder.arc(entry, fork_trans)
        for child_entry in child_entries:
            builder.arc(fork_trans, child_entry)

        return CompiledNode(
            entry=entry,
            success_exit=final_success,
            failure_exit=final_failure,
            running_exit=final_running,
            places=all_places,
            transitions=all_transitions
        )

    def _compile_decorator(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a decorator node."""
        children = node_spec.children
        if not children:
            raise ValueError(f"Decorator {node_spec.name} has no child")

        child_spec = children[0]
        decorator_name = node_spec.name.lower()

        if decorator_name.startswith("inverter"):
            return self._compile_inverter(node_spec, ctx, entry, child_spec)
        elif decorator_name.startswith("retry"):
            return self._compile_retry(node_spec, ctx, entry, child_spec)
        elif decorator_name.startswith("timeout"):
            return self._compile_timeout(node_spec, ctx, entry, child_spec)
        elif decorator_name.startswith("succeeder"):
            return self._compile_succeeder(node_spec, ctx, entry, child_spec)
        elif decorator_name.startswith("failer"):
            return self._compile_failer(node_spec, ctx, entry, child_spec)
        elif decorator_name.startswith("gate"):
            return self._compile_gate(node_spec, ctx, entry, child_spec)
        elif decorator_name.startswith("when"):
            return self._compile_when(node_spec, ctx, entry, child_spec)
        elif decorator_name.startswith("ratelimit"):
            return self._compile_rate_limit(node_spec, ctx, entry, child_spec)
        else:
            return self._compile_node(child_spec, ctx, entry)

    def _compile_inverter(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile an Inverter decorator."""
        builder = ctx.builder

        # Create swapped exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Compile the child
        compiled_child = self._compile_node(child_spec, ctx, entry)

        # Route child success to decorator failure
        builder.forward(
            compiled_child.success_exit,
            failure_exit,
            name=ctx.fresh_transition_name("invert_s_to_f")
        )

        # Route child failure to decorator success
        builder.forward(
            compiled_child.failure_exit,
            success_exit,
            name=ctx.fresh_transition_name("invert_f_to_s")
        )

        # Route child running to decorator running
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("invert_r_to_r")
            )

        return CompiledNode(
            entry=compiled_child.entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit] + compiled_child.places,
            transitions=compiled_child.transitions
        )

    def _compile_retry(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile a Retry decorator using token-carried count."""
        import re
        builder = ctx.builder

        # Extract max_attempts
        match = re.search(r'Retry\([^,]+,\s*(\d+)\)', node_spec.name)
        max_attempts = int(match.group(1)) if match else 3

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Create child entry place
        child_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_entry"))

        # Init transition
        def make_init_handler(child_entry_place, max_att):
            async def handler(consumed, bb, timebase):
                yield {child_entry_place: StatusToken(status=Status.SUCCESS, attempts_remaining=max_att)}
            return handler

        init_name = ctx.fresh_transition_name("retry_init")
        init_handler = make_init_handler(child_entry, max_attempts)
        init_handler.__name__ = init_name
        init_trans = builder.transition(delay=0.0)(init_handler)

        builder.arc(entry, init_trans)
        builder.arc(init_trans, child_entry)

        # Compile child
        compiled_child = self._compile_node(child_spec, ctx, child_entry)

        # Success -> propagate
        builder.forward(
            compiled_child.success_exit,
            success_exit,
            name=ctx.fresh_transition_name("retry_success")
        )

        # RUNNING -> propagate
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("retry_running")
            )

        # Failure -> retry check
        def make_retry_check_handler(child_entry_place, failure_exit_place):
            async def handler(consumed, bb, timebase):
                if not consumed:
                    return

                token = consumed[0]
                if not isinstance(token, StatusToken):
                    yield {failure_exit_place: StatusToken(status=Status.FAILURE)}
                    return

                if token.attempts_remaining > 0:
                    yield {
                        child_entry_place: StatusToken(
                            status=Status.SUCCESS,
                            data=token.data,
                            attempts_remaining=token.attempts_remaining - 1
                        )
                    }
                else:
                    yield {failure_exit_place: StatusToken(status=Status.FAILURE, data=token.data)}
            return handler

        check_name = ctx.fresh_transition_name("retry_check")
        check_handler = make_retry_check_handler(child_entry, failure_exit)
        check_handler.__name__ = check_name
        check_trans = builder.transition(delay=ctx.retry_delay)(check_handler)

        builder.arc(compiled_child.failure_exit, check_trans)
        builder.arc(check_trans, child_entry)
        builder.arc(check_trans, failure_exit)

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit, child_entry] + compiled_child.places,
            transitions=compiled_child.transitions + [init_trans, check_trans]
        )

    def _compile_timeout(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile a Timeout decorator using timed Petri net semantics."""
        import re
        builder = ctx.builder

        # Extract timeout
        match = re.search(r'Timeout\(([\d.]+)', node_spec.name)
        timeout_seconds = float(match.group(1)) if match else 1.0

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Create child entry and timer trigger places
        child_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_entry"))
        timer_trigger = builder.place(ctx.fresh_place_name(f"{node_spec.name}_timer_trigger"))

        # Create intermediate result places
        child_success = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_success"))
        child_failure = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_failure"))
        timeout_result = builder.place(ctx.fresh_place_name(f"{node_spec.name}_timeout_result"))

        # Compile the child
        compiled_child = self._compile_node(child_spec, ctx, child_entry)

        # Create fork transition
        def make_fork_handler(child_entry_place, timer_trigger_place):
            async def handler(consumed, bb, timebase):
                yield {
                    child_entry_place: StatusToken(status=Status.SUCCESS),
                    timer_trigger_place: StatusToken(status=Status.SUCCESS),
                }
            return handler

        fork_name = ctx.fresh_transition_name("timeout_fork")
        fork_handler = make_fork_handler(child_entry, timer_trigger)
        fork_handler.__name__ = fork_name
        fork_trans = builder.transition()(fork_handler)

        builder.arc(entry, fork_trans)
        builder.arc(fork_trans, child_entry)
        builder.arc(fork_trans, timer_trigger)

        # Wire child success/failure
        builder.forward(
            compiled_child.success_exit,
            child_success,
            name=ctx.fresh_transition_name("child_to_success")
        )
        builder.forward(
            compiled_child.failure_exit,
            child_failure,
            name=ctx.fresh_transition_name("child_to_failure")
        )

        # Wire child running
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("child_running")
            )

        # Create timer transition
        def make_timer_handler():
            async def handler(consumed, bb, timebase):
                yield {timeout_result: StatusToken(status=Status.FAILURE)}
            return handler

        timer_name = ctx.fresh_transition_name("timeout_timer")
        timer_handler = make_timer_handler()
        timer_handler.__name__ = timer_name
        timer_trans = builder.transition(delay=timeout_seconds)(timer_handler)

        builder.arc(timer_trigger, timer_trans)
        builder.arc(timer_trans, timeout_result)

        # Create merge transitions
        def make_child_success_handler(success_exit_place):
            async def handler(consumed, bb, timebase):
                yield {success_exit_place: StatusToken(status=Status.SUCCESS)}
            return handler

        child_success_merge_name = ctx.fresh_transition_name("timeout_child_success")
        child_success_handler = make_child_success_handler(success_exit)
        child_success_handler.__name__ = child_success_merge_name
        child_success_trans = builder.transition()(child_success_handler)

        builder.arc(child_success, child_success_trans)
        builder.arc(child_success_trans, success_exit)

        def make_child_failure_handler(failure_exit_place):
            async def handler(consumed, bb, timebase):
                yield {failure_exit_place: StatusToken(status=Status.FAILURE)}
            return handler

        child_failure_merge_name = ctx.fresh_transition_name("timeout_child_failure")
        child_failure_handler = make_child_failure_handler(failure_exit)
        child_failure_handler.__name__ = child_failure_merge_name
        child_failure_trans = builder.transition()(child_failure_handler)

        builder.arc(child_failure, child_failure_trans)
        builder.arc(child_failure_trans, failure_exit)

        def make_timeout_handler(failure_exit_place):
            async def handler(consumed, bb, timebase):
                yield {failure_exit_place: StatusToken(status=Status.FAILURE)}
            return handler

        timeout_merge_name = ctx.fresh_transition_name("timeout_merge")
        timeout_handler = make_timeout_handler(failure_exit)
        timeout_handler.__name__ = timeout_merge_name
        timeout_trans = builder.transition()(timeout_handler)

        builder.arc(timeout_result, timeout_trans)
        builder.arc(timeout_trans, failure_exit)

        all_transitions = compiled_child.transitions + [fork_trans, timer_trans, child_success_trans, child_failure_trans, timeout_trans]

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit, child_entry, timer_trigger,
                    child_success, child_failure, timeout_result] + compiled_child.places,
            transitions=all_transitions
        )

    def _compile_succeeder(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile a Succeeder decorator."""
        builder = ctx.builder

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Compile child
        compiled_child = self._compile_node(child_spec, ctx, entry)

        # Route all to success
        builder.forward(
            compiled_child.success_exit,
            success_exit,
            name=ctx.fresh_transition_name("succeeder_from_s")
        )
        builder.forward(
            compiled_child.failure_exit,
            success_exit,
            name=ctx.fresh_transition_name("succeeder_from_f")
        )
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                success_exit,
                name=ctx.fresh_transition_name("succeeder_from_r")
            )

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit] + compiled_child.places,
            transitions=compiled_child.transitions
        )

    def _compile_failer(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile a Failer decorator."""
        builder = ctx.builder

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Compile child
        compiled_child = self._compile_node(child_spec, ctx, entry)

        # Route all to failure
        builder.forward(
            compiled_child.success_exit,
            failure_exit,
            name=ctx.fresh_transition_name("failer_from_s")
        )
        builder.forward(
            compiled_child.failure_exit,
            failure_exit,
            name=ctx.fresh_transition_name("failer_from_f")
        )
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                failure_exit,
                name=ctx.fresh_transition_name("failer_from_r")
            )

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit] + compiled_child.places,
            transitions=compiled_child.transitions
        )

    def _compile_gate(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile a Gate decorator."""
        builder = ctx.builder

        # Get condition from payload
        payload = node_spec.payload
        if isinstance(payload, dict):
            condition_spec = payload.get("condition")
        else:
            raise ValueError(f"Gate {node_spec.name} has no condition in payload metadata")

        if condition_spec is None:
            raise ValueError(f"Gate {node_spec.name} has no condition in payload")

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Create entry place for the child
        child_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_entry"))

        # Compile condition
        compiled_condition = self._compile_node(condition_spec, ctx, entry)

        # Compile child
        compiled_child = self._compile_node(child_spec, ctx, child_entry)

        # Wire condition success -> child entry
        builder.forward(
            compiled_condition.success_exit,
            child_entry,
            name=ctx.fresh_transition_name("gate_open")
        )

        # Wire child outputs
        builder.forward(
            compiled_child.success_exit,
            success_exit,
            name=ctx.fresh_transition_name("gate_child_success")
        )
        builder.forward(
            compiled_child.failure_exit,
            failure_exit,
            name=ctx.fresh_transition_name("gate_child_failure")
        )
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("gate_child_running")
            )

        # Wire condition failure -> gate failure
        builder.forward(
            compiled_condition.failure_exit,
            failure_exit,
            name=ctx.fresh_transition_name("gate_closed")
        )

        # Wire condition running
        if compiled_condition.running_exit is not None:
            builder.forward(
                compiled_condition.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("gate_condition_running")
            )

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit, child_entry] + compiled_condition.places + compiled_child.places,
            transitions=compiled_condition.transitions + compiled_child.transitions
        )

    def _compile_when(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile a When decorator."""
        builder = ctx.builder

        # Get condition from payload
        payload = node_spec.payload
        if isinstance(payload, dict):
            condition_spec = payload.get("condition")
        else:
            raise ValueError(f"When {node_spec.name} has no condition in payload metadata")

        if condition_spec is None:
            raise ValueError(f"When {node_spec.name} has no condition in payload")

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Create entry place for the child
        child_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_entry"))

        # Compile condition
        compiled_condition = self._compile_node(condition_spec, ctx, entry)

        # Compile child
        compiled_child = self._compile_node(child_spec, ctx, child_entry)

        # Wire condition success -> child entry
        builder.forward(
            compiled_condition.success_exit,
            child_entry,
            name=ctx.fresh_transition_name("when_true")
        )

        # Wire child outputs
        builder.forward(
            compiled_child.success_exit,
            success_exit,
            name=ctx.fresh_transition_name("when_child_success")
        )
        builder.forward(
            compiled_child.failure_exit,
            failure_exit,
            name=ctx.fresh_transition_name("when_child_failure")
        )
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("when_child_running")
            )

        # Wire condition failure -> SUCCESS (skip but don't fail)
        builder.forward(
            compiled_condition.failure_exit,
            success_exit,
            name=ctx.fresh_transition_name("when_false_skip")
        )

        # Wire condition running
        if compiled_condition.running_exit is not None:
            builder.forward(
                compiled_condition.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("when_condition_running")
            )

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit, child_entry] + compiled_condition.places + compiled_child.places,
            transitions=compiled_condition.transitions + compiled_child.transitions
        )

    def _compile_rate_limit(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec"
    ) -> CompiledNode:
        """Compile a RateLimit decorator."""
        import re
        builder = ctx.builder

        # Extract period
        match = re.search(r'RateLimit\(([\d.]+)s', node_spec.name)
        period = float(match.group(1)) if match else 1.0

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Create places
        child_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_entry"))
        waiting_place = builder.place(ctx.fresh_place_name(f"{node_spec.name}_waiting"))
        child_success = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_success"))
        child_failure = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_failure"))
        child_running = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_running"))

        # Rate check transition
        def make_rate_check_handler(child_entry_place, waiting_place, period_seconds):
            async def handler(consumed, bb, timebase):
                current_time = timebase.now()

                if not consumed:
                    yield {child_entry_place: StatusToken(status=Status.SUCCESS, last_start_time=current_time, next_allowed_time=0.0)}
                    return

                token = consumed[0]
                if not isinstance(token, StatusToken):
                    yield {child_entry_place: StatusToken(status=Status.SUCCESS, last_start_time=current_time, next_allowed_time=0.0)}
                    return

                if token.next_allowed_time == 0.0 or current_time >= token.next_allowed_time:
                    yield {child_entry_place: StatusToken(status=Status.SUCCESS, last_start_time=current_time, next_allowed_time=token.next_allowed_time)}
                else:
                    yield {waiting_place: StatusToken(status=Status.RUNNING, last_start_time=token.last_start_time, next_allowed_time=token.next_allowed_time)}
            return handler

        check_name = ctx.fresh_transition_name("rate_limit_check")
        check_handler = make_rate_check_handler(child_entry, waiting_place, period)
        check_handler.__name__ = check_name
        check_trans = builder.transition()(check_handler)

        builder.arc(entry, check_trans)
        builder.arc(check_trans, child_entry)
        builder.arc(check_trans, waiting_place)

        # Compile child
        compiled_child = self._compile_node(child_spec, ctx, child_entry)

        # Wire child outputs
        builder.forward(
            compiled_child.success_exit,
            child_success,
            name=ctx.fresh_transition_name("child_success_forward")
        )
        builder.forward(
            compiled_child.failure_exit,
            child_failure,
            name=ctx.fresh_transition_name("child_failure_forward")
        )
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                child_running,
                name=ctx.fresh_transition_name("child_running_forward")
            )

        # Completion handler
        def make_completion_handler(output_place, period_seconds, output_status):
            async def handler(consumed, bb, timebase):
                if not consumed:
                    return

                token = consumed[0]
                current_time = timebase.now()

                new_token = StatusToken(
                    status=output_status,
                    last_start_time=token.last_start_time,
                    next_allowed_time=current_time + period_seconds,
                    data=token.data,
                    attempts_remaining=token.attempts_remaining,
                    child_running=token.child_running,
                    iteration_count=token.iteration_count
                )
                yield {output_place: new_token}
            return handler

        # Success completion
        success_completion_name = ctx.fresh_transition_name("rate_limit_success_completion")
        success_completion_handler = make_completion_handler(success_exit, period, Status.SUCCESS)
        success_completion_handler.__name__ = success_completion_name
        success_completion_trans = builder.transition()(success_completion_handler)

        builder.arc(child_success, success_completion_trans)
        builder.arc(success_completion_trans, success_exit)

        # Failure completion
        failure_completion_name = ctx.fresh_transition_name("rate_limit_failure_completion")
        failure_completion_handler = make_completion_handler(failure_exit, period, Status.FAILURE)
        failure_completion_handler.__name__ = failure_completion_name
        failure_completion_trans = builder.transition()(failure_completion_handler)

        builder.arc(child_failure, failure_completion_trans)
        builder.arc(failure_completion_trans, failure_exit)

        # Running handler
        def make_running_handler(running_exit_place):
            async def handler(consumed, bb, timebase):
                if not consumed:
                    return
                yield {running_exit_place: StatusToken(status=Status.RUNNING)}
            return handler

        running_forward_name = ctx.fresh_transition_name("rate_limit_running_forward")
        running_forward_handler = make_running_handler(running_exit)
        running_forward_handler.__name__ = running_forward_name
        running_forward_trans = builder.transition()(running_forward_handler)

        builder.arc(child_running, running_forward_trans)
        builder.arc(running_forward_trans, running_exit)

        # Waiting retry transition
        def make_waiting_retry(child_entry_place, period_seconds):
            async def handler(consumed, bb, timebase):
                if not consumed:
                    return
                token = consumed[0]
                current_time = timebase.now()

                if current_time >= token.next_allowed_time:
                    yield {child_entry_place: StatusToken(status=Status.SUCCESS, last_start_time=current_time, next_allowed_time=token.next_allowed_time)}
                else:
                    yield {waiting_place: StatusToken(status=Status.RUNNING, last_start_time=token.last_start_time, next_allowed_time=token.next_allowed_time)}
            return handler

        waiting_retry_name = ctx.fresh_transition_name("rate_limit_waiting_retry")
        waiting_retry_handler = make_waiting_retry(child_entry, period)
        waiting_retry_handler.__name__ = waiting_retry_name
        waiting_retry_trans = builder.transition(delay=ctx.running_delay)(waiting_retry_handler)

        builder.arc(waiting_place, waiting_retry_trans)
        builder.arc(waiting_retry_trans, child_entry)
        builder.arc(waiting_retry_trans, waiting_place)

        all_places = [entry, success_exit, failure_exit, running_exit, child_entry, waiting_place, child_success, child_failure, child_running]
        all_transitions = compiled_child.transitions + [check_trans, success_completion_trans, failure_completion_trans, running_forward_trans, waiting_retry_trans]

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=all_places + compiled_child.places,
            transitions=all_transitions
        )

    def _compile_do_while(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any,
        child_spec: "NodeSpec" = None
    ) -> CompiledNode:
        """Compile a DoWhile decorator."""
        builder = ctx.builder

        # Get condition from payload
        payload = node_spec.payload
        if isinstance(payload, dict):
            condition_spec = payload.get("condition")
        else:
            raise ValueError(f"DoWhile {node_spec.name} has no condition in payload metadata")

        if condition_spec is None:
            raise ValueError(f"DoWhile {node_spec.name} has no condition in payload")

        # Get child from children list
        if child_spec is None:
            children = node_spec.children
            if not children:
                raise ValueError(f"DoWhile {node_spec.name} has no child")
            child_spec = children[0]

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Create places
        condition_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_condition_entry"))
        child_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_child_entry"))
        loop_back_place = builder.place(ctx.fresh_place_name(f"{node_spec.name}_loop_back"))

        # Init transition
        def make_init_handler(condition_entry_place):
            async def handler(consumed, bb, timebase):
                yield {condition_entry_place: StatusToken(status=Status.SUCCESS, iteration_count=0)}
            return handler

        init_name = ctx.fresh_transition_name("do_while_init")
        init_handler = make_init_handler(condition_entry)
        init_handler.__name__ = init_name
        init_trans = builder.transition()(init_handler)

        builder.arc(entry, init_trans)
        builder.arc(init_trans, condition_entry)

        # Compile condition
        compiled_condition = self._compile_node(condition_spec, ctx, condition_entry)

        # Condition success dispatcher
        def make_condition_dispatcher(child_entry_place, success_exit_place):
            async def handler(consumed, bb, timebase):
                if not consumed:
                    return

                token = consumed[0]
                if not isinstance(token, StatusToken):
                    yield {child_entry_place: StatusToken(status=Status.SUCCESS)}
                    return

                yield {child_entry_place: StatusToken(status=Status.SUCCESS, iteration_count=token.iteration_count)}
            return handler

        cond_dispatcher_name = ctx.fresh_transition_name("do_while_cond_success")
        cond_dispatcher_handler = make_condition_dispatcher(child_entry, success_exit)
        cond_dispatcher_handler.__name__ = cond_dispatcher_name
        cond_dispatcher_trans = builder.transition()(cond_dispatcher_handler)

        builder.arc(compiled_condition.success_exit, cond_dispatcher_trans)
        builder.arc(cond_dispatcher_trans, child_entry)

        # Condition failure -> SUCCESS
        builder.forward(
            compiled_condition.failure_exit,
            success_exit,
            name=ctx.fresh_transition_name("do_while_complete")
        )

        # Condition running -> running exit
        if compiled_condition.running_exit is not None:
            builder.forward(
                compiled_condition.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("do_while_cond_running")
            )

        # Compile child
        compiled_child = self._compile_node(child_spec, ctx, child_entry)

        # Child success -> loop back
        def make_loop_back_handler(condition_entry_place, running_exit_place):
            async def handler(consumed, bb, timebase):
                if not consumed:
                    return

                token = consumed[0]
                iteration = token.iteration_count if isinstance(token, StatusToken) else 0

                yield {
                    condition_entry_place: StatusToken(status=Status.SUCCESS, iteration_count=iteration + 1),
                    running_exit_place: StatusToken(status=Status.RUNNING)
                }
            return handler

        loop_back_name = ctx.fresh_transition_name("do_while_loop_back")
        loop_back_handler = make_loop_back_handler(condition_entry, running_exit)
        loop_back_handler.__name__ = loop_back_name
        loop_back_trans = builder.transition(delay=ctx.running_delay)(loop_back_handler)

        builder.arc(compiled_child.success_exit, loop_back_trans)
        builder.arc(loop_back_trans, condition_entry)
        builder.arc(loop_back_trans, running_exit)

        # Child failure -> failure exit
        builder.forward(
            compiled_child.failure_exit,
            failure_exit,
            name=ctx.fresh_transition_name("do_while_child_fail")
        )

        # Child running -> running exit
        if compiled_child.running_exit is not None:
            builder.forward(
                compiled_child.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("do_while_child_running")
            )

        all_transitions = (compiled_condition.transitions + compiled_child.transitions +
                          [init_trans, cond_dispatcher_trans, loop_back_trans])

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit, condition_entry, child_entry, loop_back_place] +
                    compiled_condition.places + compiled_child.places,
            transitions=all_transitions
        )

    def _compile_try_catch(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a TryCatch decorator."""
        builder = ctx.builder

        # Get try and catch specs from payload
        try_spec = node_spec.payload.get("try")
        catch_spec = node_spec.payload.get("catch")

        if try_spec is None:
            raise ValueError(f"TryCatch {node_spec.name} has no try block in payload")
        if catch_spec is None:
            raise ValueError(f"TryCatch {node_spec.name} has no catch block in payload")

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Create entry place for catch block
        catch_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_catch_entry"))

        # Compile try block
        compiled_try = self._compile_node(try_spec, ctx, entry)

        # Wire try success -> try_catch success
        builder.forward(
            compiled_try.success_exit,
            success_exit,
            name=ctx.fresh_transition_name("try_catch_try_success")
        )

        # Wire try failure -> catch entry
        builder.forward(
            compiled_try.failure_exit,
            catch_entry,
            name=ctx.fresh_transition_name("try_catch_to_catch")
        )

        # Wire try running
        if compiled_try.running_exit is not None:
            builder.forward(
                compiled_try.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("try_catch_try_running")
            )

        # Compile catch block
        compiled_catch = self._compile_node(catch_spec, ctx, catch_entry)

        # Wire catch success -> try_catch success
        builder.forward(
            compiled_catch.success_exit,
            success_exit,
            name=ctx.fresh_transition_name("try_catch_catch_success")
        )

        # Wire catch failure -> try_catch failure
        builder.forward(
            compiled_catch.failure_exit,
            failure_exit,
            name=ctx.fresh_transition_name("try_catch_catch_failure")
        )

        # Wire catch running
        if compiled_catch.running_exit is not None:
            builder.forward(
                compiled_catch.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("try_catch_catch_running")
            )

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit, catch_entry] + compiled_try.places + compiled_catch.places,
            transitions=compiled_try.transitions + compiled_catch.transitions
        )

    def _compile_subtree(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a subtree node (modular BT inclusion)."""
        builder = ctx.builder

        # Extract the subtree's root from payload
        subtree_root = node_spec.payload.get("root")
        if subtree_root is None:
            raise ValueError(f"Subtree {node_spec.name} has no root in payload")

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # Compile the subtree's root as a child
        compiled_subtree = self._compile_node(subtree_root, ctx, entry)

        # Wire the subtree's exits to our exits
        builder.forward(
            compiled_subtree.success_exit,
            success_exit,
            name=ctx.fresh_transition_name("subtree_success")
        )
        builder.forward(
            compiled_subtree.failure_exit,
            failure_exit,
            name=ctx.fresh_transition_name("subtree_failure")
        )
        if compiled_subtree.running_exit is not None:
            builder.forward(
                compiled_subtree.running_exit,
                running_exit,
                name=ctx.fresh_transition_name("subtree_running")
            )

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=[entry, success_exit, failure_exit, running_exit] + compiled_subtree.places,
            transitions=compiled_subtree.transitions
        )

    def _compile_match(
        self,
        node_spec: "NodeSpec",
        ctx: CompilationContext,
        entry: Any
    ) -> CompiledNode:
        """Compile a Match decorator with pattern-matching dispatch."""
        from ..rhizomorph.core import _DefaultCase

        builder = ctx.builder

        # Extract key_fn and cases from payload
        key_fn = node_spec.payload.get("key_fn")
        case_specs = node_spec.payload.get("cases", [])

        if not case_specs:
            raise ValueError(f"Match {node_spec.name} has no cases")

        # Create exits
        success_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_success"))
        failure_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_failure"))
        running_exit = builder.place(ctx.fresh_place_name(f"{node_spec.name}_running"))

        # No match place
        no_match = builder.place(ctx.fresh_place_name(f"{node_spec.name}_no_match"))

        # Create entry places and compile each case child
        case_entries = []
        all_places = [entry, success_exit, failure_exit, running_exit, no_match]
        all_transitions = []

        for case_spec in case_specs:
            case_entry = builder.place(ctx.fresh_place_name(f"{node_spec.name}_case_{len(case_entries)}_entry"))
            case_entries.append(case_entry)
            all_places.append(case_entry)

            # Compile the child for this case
            compiled_child = self._compile_node(case_spec.child, ctx, case_entry)
            all_places.extend(compiled_child.places)
            all_transitions.extend(compiled_child.transitions)

            # Forward child outputs
            builder.forward(
                compiled_child.success_exit,
                success_exit,
                name=ctx.fresh_transition_name("case_success")
            )
            builder.forward(
                compiled_child.failure_exit,
                failure_exit,
                name=ctx.fresh_transition_name("case_failure")
            )
            if compiled_child.running_exit is not None:
                builder.forward(
                    compiled_child.running_exit,
                    running_exit,
                    name=ctx.fresh_transition_name("case_running")
                )

        # Dispatch transition
        def make_dispatch_handler(key_fn, case_specs, case_entries, no_match_place):
            async def handler(consumed, bb, timebase):
                # Check for existing match
                matched_idx = None
                if consumed:
                    token = consumed[0]
                    if isinstance(token, MatchToken):
                        matched_idx = token.matched_case_idx
                    elif isinstance(token, StatusToken):
                        pass

                # If already matched, route directly
                if matched_idx is not None:
                    yield {case_entries[matched_idx]: token}
                    return

                # First tick: evaluate key_fn
                key_value = key_fn(bb)

                for i, case_spec in enumerate(case_specs):
                    if _matches(case_spec.matcher, key_value):
                        base_token = consumed[0] if consumed else StatusToken(status=Status.SUCCESS)
                        match_token = MatchToken(
                            base=base_token,
                            matched_case_idx=i,
                            key_value=key_value
                        )
                        yield {case_entries[i]: match_token}
                        return

                # No match
                base_token = consumed[0] if consumed else StatusToken(status=Status.FAILURE)
                yield {no_match_place: base_token}

            return handler

        dispatch_name = ctx.fresh_transition_name("match_dispatch")
        dispatch_handler = make_dispatch_handler(key_fn, case_specs, case_entries, no_match)
        dispatch_handler.__name__ = dispatch_name
        dispatch_trans = builder.transition()(dispatch_handler)
        all_transitions.append(dispatch_trans)

        builder.arc(entry, dispatch_trans)
        for case_entry in case_entries:
            builder.arc(dispatch_trans, case_entry)
        builder.arc(dispatch_trans, no_match)

        # No match -> failure
        builder.forward(no_match, failure_exit, name="no_match_to_failure")

        return CompiledNode(
            entry=entry,
            success_exit=success_exit,
            failure_exit=failure_exit,
            running_exit=running_exit,
            places=all_places,
            transitions=all_transitions
        )


def _matches(matcher: Any, value: Any) -> bool:
    """Check if a matcher matches the given value."""
    from ..rhizomorph.core import _DefaultCase

    if matcher is _DefaultCase:
        return True
    if isinstance(matcher, type):
        return isinstance(value, matcher)
    if callable(matcher):
        return bool(matcher(value))
    return value == matcher


def compile_bt_to_pn(
    tree: Any,
    name: str = "CompiledBT",
    running_delay: float = 0.1,
    retry_delay: float = 0.0
) -> NetSpec:
    """
    Compile a behavior tree to a Petri net specification.

    This is the main entry point for compiling Rhizomorph behavior trees
    to Hypha Petri nets within the Mycelium framework.

    Args:
        tree: The behavior tree to compile (a @bt.tree decorated function)
        name: Name for the compiled Petri net (default: "CompiledBT")
        running_delay: Delay between RUNNING retry attempts in seconds (default: 0.1)
        retry_delay: Delay between Retry decorator attempts in seconds (default: 0.0)

    Returns:
        NetSpec that can be executed by Hypha's PNRunner

    Example:
        >>> from mycorrhizal.mycelium import compile_bt_to_pn, PNRunner
        >>> from mycorrhizal.rhizomorph.core import bt, Status
        >>>
        >>> @bt.tree
        >>> def MyBT():
        >>>     @bt.action
        >>>     async def my_action(bb):
        >>>         return Status.SUCCESS
        >>>
        >>>     @bt.root
        >>>     @bt.sequence
        >>>     def root():
        >>>         yield my_action
        >>>
        >>> # Compile and run
        >>> pn_spec = compile_bt_to_pn(MyBT)
        >>> runner = PNRunner(pn_spec, blackboard=bb)
        >>> await runner.start(timebase)
    """
    # Get the root NodeSpec from the tree
    if hasattr(tree, 'root'):
        tree_spec = tree.root
    elif hasattr(tree, '_spec'):
        tree_spec = tree._spec
    else:
        raise ValueError(
            f"Invalid tree object: {tree}. "
            f"Expected a @bt.tree decorated function or BT namespace."
        )

    # Create compiler and compile
    compiler = BTtoPNCompiler(
        name=name,
        running_delay=running_delay,
        retry_delay=retry_delay
    )
    return compiler.compile(tree_spec)
