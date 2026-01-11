#!/usr/bin/env python3
"""
Hypha DSL - Builder Layer

NetBuilder provides the API for declaratively constructing Petri net specifications.
Used within @pn.net decorated functions.
"""

from typing import Optional, Callable, List, Dict, Type, Any, Union
from .specs import (
    NetSpec,
    PlaceSpec,
    TransitionSpec,
    GuardSpec,
    ArcSpec,
    PlaceType,
    PlaceRef,
    TransitionRef,
    SubnetRef,
)


class ArcChain:
    """Fluent interface for chaining arc definitions"""

    def __init__(self, builder: "NetBuilder", last_ref: Any):
        self.builder = builder
        self.last_ref = last_ref
        self.last_type = type(last_ref)

    def arc(
        self, target: Any, name: Optional[str] = None, weight: int = 1
    ) -> "ArcChain":
        """Chain another arc from the last element to target"""
        target_type = type(target)

        if self.last_type == target_type:
            raise ValueError(
                f"Cannot connect {self.last_type.__name__} to {target_type.__name__} directly. "
                f"Arcs must alternate between places and transitions."
            )

        arc_spec = ArcSpec(self.last_ref, target, weight, name)
        self.builder.spec.arcs.append(arc_spec)

        return ArcChain(self.builder, target)


class NetBuilder:
    """Builder for constructing Petri net specifications"""

    def __init__(self, name: str, parent: Optional[NetSpec] = None):
        self.spec = NetSpec(name, parent=parent)

    def place(
        self,
        name_or_func: Union[str, Callable, None] = None,
        type: PlaceType = PlaceType.BAG,
        state_factory: Optional[Callable] = None,
    ) -> Union[PlaceRef, Callable]:
        """Declare a regular place.

        Can be used in three ways:

        1. As a method call (returns PlaceRef for use in arcs):
           queue = builder.place("queue", type=PlaceType.QUEUE)

        2. As a decorator without parens (infers name from function):
           @builder.place
           def queue(bb):
               return bb.tokens

        3. As a decorator with custom name:
           @builder.place("custom_name")
           def my_func(bb):
               return bb.tokens
        """
        # Case 1: Used as decorator without parens - first arg is the function
        if callable(name_or_func):
            func = name_or_func
            place_name = func.__name__
            place_spec = PlaceSpec(place_name, type, handler=func, state_factory=state_factory)
            self.spec.places[place_name] = place_spec
            return PlaceRef(place_name, self.spec)

        # Case 2 & 3: Used with explicit name (as method call or decorator with args)
        # or called without any args (need to return decorator)
        if isinstance(name_or_func, str):
            name = name_or_func
            place_spec = PlaceSpec(name, type, state_factory=state_factory)
            self.spec.places[name] = place_spec
            # Return PlaceRef - it's callable via __call__ for decorator use
            return PlaceRef(name, self.spec)

        # Case 4: Called without any args - shouldn't happen but handle gracefully
        # This would be like @builder.place() with empty parens
        def decorator(func: Callable) -> PlaceRef:
            place_name = func.__name__
            place_spec = PlaceSpec(place_name, type, handler=func, state_factory=state_factory)
            self.spec.places[place_name] = place_spec
            return PlaceRef(place_name, self.spec)
        return decorator

    def io_input_place(self):
        """Decorator for IOInputPlace with async generator"""

        def decorator(func: Callable) -> PlaceRef:
            name = func.__name__
            place_spec = PlaceSpec(
                name, PlaceType.QUEUE, handler=func, is_io_input=True
            )
            self.spec.places[name] = place_spec
            return PlaceRef(name, self.spec)

        return decorator

    def io_output_place(self):
        """Decorator for IOOutputPlace with async handler"""

        def decorator(func: Callable) -> PlaceRef:
            name = func.__name__
            place_spec = PlaceSpec(
                name, PlaceType.QUEUE, handler=func, is_io_output=True
            )
            self.spec.places[name] = place_spec
            return PlaceRef(name, self.spec)

        return decorator

    def guard(self, func: Callable) -> GuardSpec:
        """Create a guard specification from a function"""
        return GuardSpec(func)

    def transition(
        self,
        guard: Optional[GuardSpec] = None,
        state_factory: Optional[Callable] = None,
    ):
        """Decorator for transition function"""

        def decorator(func: Callable) -> TransitionRef:
            name = func.__name__
            trans_spec = TransitionSpec(name, func, guard, state_factory)
            self.spec.transitions[name] = trans_spec
            return TransitionRef(name, self.spec)

        return decorator

    def arc(
        self, source: Any, target: Any, name: Optional[str] = None, weight: int = 1
    ) -> ArcChain:
        """Create an arc and return chainable ArcChain"""
        source_type = type(source)
        target_type = type(target)

        if source_type == target_type:
            raise ValueError(
                f"Cannot connect {source_type.__name__} to {target_type.__name__} directly. "
                f"Arcs must alternate between places and transitions."
            )

        arc_spec = ArcSpec(source, target, weight, name)
        self.spec.arcs.append(arc_spec)

        return ArcChain(self, target)

    def subnet(self, net_func: Callable, instance_name: str) -> SubnetRef:
        """Instantiate a subnet with given instance name"""
        # 1. Validation
        if not hasattr(net_func, "_spec"):
            raise ValueError(
                f"{net_func.__name__} is not a valid net. "
                f"Did you forget to decorate it with @pn.net?"
            )

        original_spec = net_func._spec

        # 2. Create subnet spec and copy places/transitions
        subnet_spec = NetSpec(instance_name, parent=self.spec)

        # Copy places and transitions (PlaceSpec/TransitionSpec are plain
        # dataclasses holding callables/state factories; keeping the same
        # instances is acceptable since they are stateless descriptors).
        subnet_spec.places = dict(original_spec.places)
        subnet_spec.transitions = dict(original_spec.transitions)

        # 3. Remap arcs to point into the new subnet
        subnet_spec.arcs = self._remap_arcs_to_subnet(original_spec.arcs, subnet_spec)

        # 4. Ensure nested subnets are copied
        self._ensure_nested_subnets_copied(original_spec, subnet_spec)

        # 5. Register and return
        self.spec.subnets[instance_name] = subnet_spec
        return SubnetRef(subnet_spec)

    def _remap_arcs_to_subnet(self, arcs: List[ArcSpec], subnet_spec: NetSpec) -> List[ArcSpec]:
        """Remap arc references to point into target subnet.

        For each PlaceRef/TransitionRef in the arcs, creates a new reference
        bound to the target subnet. Handles nested SubnetRefs by recursively
        copying their specs into the target subnet.

        Args:
            arcs: Original arcs from the template net
            subnet_spec: Target subnet spec to bind references to

        Returns:
            New list of ArcSpec with remapped references
        """
        new_arcs = []
        for arc in arcs:
            new_src = self._remap_ref_to_subnet(arc.source, subnet_spec, original_spec=None)
            new_tgt = self._remap_ref_to_subnet(arc.target, subnet_spec, original_spec=None)
            new_arcs.append(ArcSpec(new_src, new_tgt, arc.weight, arc.name))
        return new_arcs

    def _remap_ref_to_subnet(self, ref: Any, target_subnet: NetSpec, original_spec: Optional[NetSpec] = None) -> Any:
        """Remap a single reference to point into target subnet.

        Handles:
        - PlaceRef: Create new PlaceRef bound to target_subnet
        - TransitionRef: Create new TransitionRef bound to target_subnet
        - SubnetRef: Recursively copy nested subnet spec into target_subnet

        Args:
            ref: Reference to remap (PlaceRef, TransitionRef, SubnetRef, or other)
            target_subnet: The subnet spec to bind the reference to
            original_spec: The original template spec (for resolving nested subnets)

        Returns:
            The remapped reference, or original if unknown type
        """
        from .specs import PlaceRef, TransitionRef, SubnetRef

        if isinstance(ref, PlaceRef):
            return PlaceRef(ref.local_name, target_subnet)
        if isinstance(ref, TransitionRef):
            return TransitionRef(ref.local_name, target_subnet)
        if isinstance(ref, SubnetRef):
            # Handle nested subnets recursively
            nested_name = ref.spec.name
            if nested_name not in target_subnet.subnets:
                # Copy the nested spec into this subnet (recursive)
                copied_nested = self._copy_spec_with_parent(ref.spec, target_subnet)
                target_subnet.subnets[nested_name] = copied_nested
                return SubnetRef(copied_nested)
            else:
                # Already copied, return reference to existing copy
                return SubnetRef(target_subnet.subnets[nested_name])

        # Unknown ref type: return as-is
        return ref

    def _ensure_nested_subnets_copied(self, original_spec: NetSpec, subnet_spec: NetSpec):
        """Ensure all nested subnets are copied into the target subnet.

        This handles nested subnets that weren't already processed during
        arc remapping. Each nested subnet is copied with its parent set
        to the target subnet.

        Args:
            original_spec: The template spec containing nested subnets
            subnet_spec: The target subnet spec to copy nested subnets into
        """
        for sub_name, sub_spec in original_spec.subnets.items():
            # Only copy if not already handled (e.g., by arc remapping)
            if sub_name not in subnet_spec.subnets:
                subnet_spec.subnets[sub_name] = self._copy_spec_with_parent(
                    sub_spec, subnet_spec
                )

    def _copy_spec_with_parent(self, original: NetSpec, new_parent: NetSpec) -> NetSpec:
        """Deep copy a spec and set new parent"""
        new_spec = NetSpec(original.name, parent=new_parent)
        new_spec.places = dict(original.places)
        new_spec.transitions = dict(original.transitions)
        new_spec.arcs = list(original.arcs)

        for sub_name, sub_spec in original.subnets.items():
            new_spec.subnets[sub_name] = self._copy_spec_with_parent(sub_spec, new_spec)

        return new_spec

    def forward(self, input_place: Any, output_place: Any, name: Optional[str] = None):
        """Create a simple pass-through transition"""
        trans_name = (
            name or f"forward_{input_place.local_name}_to_{output_place.local_name}"
        )

        def make_handler(out_place):
            async def forward_handler(consumed, bb, timebase):
                for token in consumed:
                    yield {out_place: token}

            return forward_handler

        trans_spec = TransitionSpec(trans_name, make_handler(output_place))
        self.spec.transitions[trans_name] = trans_spec
        trans_ref = TransitionRef(trans_name, self.spec)

        self.spec.arcs.append(ArcSpec(input_place, trans_ref))
        self.spec.arcs.append(ArcSpec(trans_ref, output_place))

    def fork(
        self, input_place: Any, output_places: List[Any], name: Optional[str] = None
    ):
        """Create a transition that broadcasts tokens to multiple outputs"""
        trans_name = name or f"fork_{input_place.local_name}"

        def make_handler(out_places):
            async def fork_handler(consumed, bb, timebase):
                for token in consumed:
                    for output_place in out_places:
                        yield {output_place: token}

            return fork_handler

        trans_spec = TransitionSpec(trans_name, make_handler(output_places))
        self.spec.transitions[trans_name] = trans_spec
        trans_ref = TransitionRef(trans_name, self.spec)

        self.spec.arcs.append(ArcSpec(input_place, trans_ref))
        for output_place in output_places:
            self.spec.arcs.append(ArcSpec(trans_ref, output_place))

    def join(
        self, input_places: List[Any], output_place: Any, name: Optional[str] = None
    ):
        """Create a transition that waits for tokens from all inputs"""
        trans_name = name or f"join_to_{output_place.local_name}"

        def make_handler(out_place):
            async def join_handler(consumed, bb, timebase):
                for token in consumed:
                    yield {out_place: token}

            return join_handler

        trans_spec = TransitionSpec(trans_name, make_handler(output_place))
        self.spec.transitions[trans_name] = trans_spec
        trans_ref = TransitionRef(trans_name, self.spec)

        for input_place in input_places:
            self.spec.arcs.append(ArcSpec(input_place, trans_ref))
        self.spec.arcs.append(ArcSpec(trans_ref, output_place))

    def merge(
        self, input_places: List[Any], output_place: Any, name: Optional[str] = None
    ):
        """Create transitions from each input to output"""
        for i, input_place in enumerate(input_places):
            trans_name = (
                name or f"{input_place.local_name}_to_{output_place.local_name}"
            )
            if len(input_places) > 1 and not name:
                trans_name = (
                    f"{input_place.local_name}_{i}_to_{output_place.local_name}"
                )

            def make_handler(out_place):
                async def merge_handler(consumed, bb, timebase):
                    for token in consumed:
                        yield {out_place: token}

                return merge_handler

            trans_spec = TransitionSpec(trans_name, make_handler(output_place))
            self.spec.transitions[trans_name] = trans_spec
            trans_ref = TransitionRef(trans_name, self.spec)

            self.spec.arcs.append(ArcSpec(input_place, trans_ref))
            self.spec.arcs.append(ArcSpec(trans_ref, output_place))

    def round_robin(
        self, input_place: Any, output_places: List[Any], name: Optional[str] = None
    ):
        """Create a transition that distributes tokens round-robin"""
        trans_name = name or f"round_robin_{input_place.local_name}"

        def make_handler(out_places):
            async def round_robin_handler(consumed, bb, timebase, state):
                for token in consumed:
                    output_place = out_places[state["index"]]
                    state["index"] = (state["index"] + 1) % len(out_places)
                    yield {output_place: token}

            return round_robin_handler

        trans_spec = TransitionSpec(
            trans_name, make_handler(output_places), state_factory=lambda: {"index": 0}
        )
        self.spec.transitions[trans_name] = trans_spec
        trans_ref = TransitionRef(trans_name, self.spec)

        self.spec.arcs.append(ArcSpec(input_place, trans_ref))
        for output_place in output_places:
            self.spec.arcs.append(ArcSpec(trans_ref, output_place))

    def route(
        self, input_place: Any, type_map: Dict[Type, Any], name: Optional[str] = None
    ):
        """Create a transition that routes by token type"""
        trans_name = name or f"route_{input_place.local_name}"

        def make_handler(t_map):
            async def route_handler(consumed, bb, timebase):
                for token in consumed:
                    token_type = type(token)
                    if token_type in t_map:
                        yield {t_map[token_type]: token}

            return route_handler

        trans_spec = TransitionSpec(trans_name, make_handler(type_map))
        self.spec.transitions[trans_name] = trans_spec
        trans_ref = TransitionRef(trans_name, self.spec)

        self.spec.arcs.append(ArcSpec(input_place, trans_ref))
        for output_place in type_map.values():
            self.spec.arcs.append(ArcSpec(trans_ref, output_place))


class PetriNetDSL:
    """Module-level API for Petri net definition"""

    @staticmethod
    def net(func: Callable) -> Callable:
        """
        Decorator for defining a Petri net.
        The decorated function receives a NetBuilder as its parameter.
        """
        builder = NetBuilder(func.__name__, parent=None)
        func(builder)

        func._spec = builder.spec
        func.to_mermaid = lambda: builder.spec.to_mermaid()

        return func


pn = PetriNetDSL()
