#!/usr/bin/env python3
"""
Cordyceps Asyncio Petri Net Utilities: Parametric Fork and Join Interfaces

Updated for the autonomous asyncio architecture with PlaceName enum support.
"""
from .core import (
    Interface,
    Place,
    Transition,
    Arc,
    PlaceName,
    PetriNet,
    _decl_stack_get,
    DeclarativeMeta,
    _DeclarativeNamespace
)
from typing import NamedTuple, Type, Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import asyncio
from asyncio import sleep
from inspect import FrameInfo
import inspect
from abc import ABC, ABCMeta
from typing import Optional, NamedTuple
from types import FrameType


def _inject(name: str, obj: object, target_depth: int = 0) -> bool:
    """
    Inject an object into the active declarative class body's namespace.
    target_depth=0 -> innermost class being defined; larger values walk outward.
    """
    stack = _decl_stack_get()
    if not stack:
        return False
    if target_depth < 0:
        # Allow negative indexing semantics if desired
        target_depth = len(stack) + target_depth
    if not (0 <= target_depth < len(stack)):
        return False
    target_ns = stack[-1 - target_depth]  # innermost at end
    target_ns[name] = obj
    return True


def _inject_into_innermost(name: str, obj: object) -> bool:
    return _inject(name, obj, target_depth=0)


def _inject_into_outermost(name: str, obj: object) -> bool:
    stack = _decl_stack_get()
    if not stack:
        return False
    return _inject(name, obj, target_depth=len(stack) - 1)


def _find_declarative_container_frame(ns: _DeclarativeNamespace) -> Optional[FrameType]:
    """
    Walk up the call stack to find the *class-body* frame whose locals mapping
    is exactly the prepared namespace `ns`. That frame's f_lineno corresponds to
    the line in the declarative container where the helper was invoked.
    """
    frame = inspect.currentframe()
    try:
        # Skip this function's own frame
        if frame is not None:
            frame = frame.f_back

        while frame is not None:
            # For class bodies, CPython executes into the mapping returned by __prepare__
            # so f_locals is *that* mapping. We match by identity.
            if frame.f_locals is ns:
                return frame

            # Be defensive: if a different instance of our mapping type leaks in,
            # also accept frames whose locals look like a prepared namespace.
            locs = frame.f_locals
            if isinstance(locs, _DeclarativeNamespace) and locs is ns:
                return frame

            frame = frame.f_back
    finally:
        # avoid reference cycles
        del frame

    return None


def _gen_prefix() -> str:
    """
    Generate a prefix using the *declarative container's* current line number,
    not the helper's, plus a per-namespace counter for disambiguation.

    Format: <qualname>_L<lineno>_<counter>
    """
    stack = _decl_stack_get()
    if not stack:
        raise RuntimeError("No declarative container found.")

    ns = stack[-1]  # innermost container
    qualname = ns.get("__qualname__", "Anonymous")

    # Find the class-body frame that owns this namespace mapping
    container_frame = _find_declarative_container_frame(ns)
    lineno = container_frame.f_lineno if container_frame is not None else 0

    # Per-namespace monotonic counter (handles multiple helpers on same line)
    counter_key = "__decl_counter__"
    next_index = int(ns.get(counter_key, 0)) + 1
    ns[counter_key] = next_index

    return f"{qualname}_L{lineno}_{next_index}"


def ForkN(n: int, input: Type[Place]) -> List[Type[Place]]:
    """
    Returns an Interface subclass with 1 input place, n output places, and a transition
    that replicates each input token to all outputs, using the Fork helper.
    """

    # Create output places
    output_classes = []
    prefix = _gen_prefix()
    for i in range(n):
        output_cls = type(f"{prefix}_Output_{i+1}", (Place,), {})
        output_classes.append(output_cls)

    # Use Fork to create the transition
    Fork(input, output_classes)

    return output_classes


def Fork(from_place_class: Type[Place], to_place_classes: List[Type[Place]]):
    """
    Declaratively create a fork transition from one place to multiple places.
    Usage: Fork(MyInterface.InputPlace, [MyInterface.Output1, MyInterface.Output2])
    """
    # Generate enum mapping automatically
    place_names = ["INPUT"] + [f"OUTPUT_{i+1}" for i in range(len(to_place_classes))]
    ForkPlaceNames = Enum(
        "ForkPlaceNames", {name: name.lower() for name in place_names}, type=PlaceName
    )
    prefix = _gen_prefix()

    # Create mapping
    enum_mapping = {from_place_class: ForkPlaceNames.INPUT}
    for i, place_class in enumerate(to_place_classes):
        enum_key = getattr(ForkPlaceNames, f"OUTPUT_{i+1}")
        enum_mapping[place_class] = enum_key

    class ForkTransition(Transition):
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[from_place_class]: Arc(from_place_class)}

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                enum_mapping[place_class]: Arc(place_class)
                for place_class in to_place_classes
            }

        async def on_fire(
            self, consumed: Dict[PlaceName, List]
        ) -> Dict[PlaceName, List]:
            result = {enum_mapping[place_class]: [] for place_class in to_place_classes}

            input_tokens = consumed[enum_mapping[from_place_class]]
            for token in input_tokens:
                for place_class in to_place_classes:
                    result[enum_mapping[place_class]].append(token)

            return result

    _inject_into_innermost(f"{prefix}_ForkTransition", ForkTransition)


def Forward(input: Type[Place], output: Type[Place]):
    """
    Declaratively create a forward transition from one place to another.
    Usage: Forward(MyInterface.Input, MyInterface.Output)
    """

    class ForwardPlaceName(PlaceName):
        INPUT = "input"
        OUTPUT = "output"

    class ForwardTransition(Transition):
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {ForwardPlaceName.INPUT: Arc(input)}

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {ForwardPlaceName.OUTPUT: Arc(output)}

    prefix = _gen_prefix()
    _inject_into_innermost(f"{prefix}_ForwardTransition", ForwardTransition)


def ForwardFrom(input: Type[Place]) -> Type[Place]:
    """
    Returns a tuple: (transition, output_place_class), using the Forward helper.
    Usage: transition, output_place = ForwardFrom(Input)
    """
    prefix = _gen_prefix()
    output_place = type(f"{prefix}_Output", (Place,), {})
    Forward(input, output_place)
    return output_place


def Join(from_place_classes: List[Type[Place]], to_place_class: Type[Place]):
    """
    Declaratively create a join transition from multiple places to one place.
    Usage: Join([MyInterface.Input1, MyInterface.Input2], MyInterface.Output)
    Returns a Transition subclass that requires tokens from all input places before firing to the output.
    """
    prefix = _gen_prefix()

    # Generate enum mapping automatically
    place_names = [f"INPUT_{i+1}" for i in range(len(from_place_classes))] + ["OUTPUT"]
    JoinPlaceNames = Enum(
        "JoinPlaceNames", {name: name.lower() for name in place_names}, type=PlaceName
    )

    # Create mapping
    enum_mapping = {to_place_class: JoinPlaceNames.OUTPUT}
    for i, place_class in enumerate(from_place_classes):
        enum_key = getattr(JoinPlaceNames, f"INPUT_{i+1}")
        enum_mapping[place_class] = enum_key

    class JoinTransition(Transition):
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {
                enum_mapping[place_class]: Arc(place_class)
                for place_class in from_place_classes
            }

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[to_place_class]: Arc(to_place_class)}

        async def on_fire(
            self, consumed: Dict[PlaceName, List]
        ) -> Dict[PlaceName, List]:
            # Only fire if all input places have tokens
            all_tokens = []
            for place_class in from_place_classes:
                tokens = consumed[enum_mapping[place_class]]
                all_tokens.extend(tokens)
            return {enum_mapping[to_place_class]: all_tokens}

    _inject_into_innermost(f"{prefix}_JoinTransition", JoinTransition)


def JoinFrom(inputs: List[Type[Place]]) -> Type[Place]:
    """
    Returns a tuple: (transition, output_place_class), using the Join helper.
    Usage: transition, output_place = JoinFrom([Input1, Input2])
    """
    prefix = _gen_prefix()
    output_place = type(f"{prefix}_Output", (Place,), {})
    Join(inputs, output_place)
    return output_place


def Merge(inputs: List[Type[Place]], output: Type[Place]):
    """
    Declaratively create a merge from multiple places to one place.
    Usage: Merge([MyInterface.Input1, MyInterface.Input2], MyInterface.Output)
    Returns a list of transitions, each drawing from an input place to the single output place.
    """
    # Generate enum mapping automatically
    place_names = [f"INPUT_{i+1}" for i in range(len(inputs))] + ["OUTPUT"]
    MergePlaceNames = Enum(
        "MergePlaceNames", {name: name.lower() for name in place_names}, type=PlaceName
    )

    # Create one transition per input place, each feeding into the output place
    def create_transition(idx):
        class _MergeTransition(Transition):
            def input_arcs(self) -> Dict[PlaceName, Arc]:
                enum_key = getattr(MergePlaceNames, f"INPUT_{idx+1}")
                return {enum_key: Arc(inputs[idx])}

            def output_arcs(self) -> Dict[PlaceName, Arc]:
                return {MergePlaceNames.OUTPUT: Arc(output)}

        return _MergeTransition

    transitions = []
    prefix = _gen_prefix()
    for i in range(len(inputs)):
        created_transition = create_transition(i)
        # Rename the transition class
        _inject_into_innermost(f"{prefix}_MergeTransition_{i+1}", created_transition)


def MergeFrom(inputs: List[Type[Place]]):
    """
    Create a merge interface from multiple input places to a single new output place.
    Usage: class MyMerge(MergeFrom([Input1, Input2])): pass
    """
    prefix = _gen_prefix()
    output = type(f"{prefix}_Output", (Place,), {})
    Merge(inputs, output)


def Buffer(capacity: int) -> Type[Interface]:
    """
    Create a buffer interface with specified capacity.
    Useful for rate limiting and backpressure management.

    Usage: class MyBuffer(Buffer(capacity=10)): pass
    """
    BufferPlaceNames = Enum(
        "BufferPlaceNames",
        {"INPUT": "input", "BUFFER": "buffer", "OUTPUT": "output"},
        type=PlaceName,
    )

    attrs = {}

    class Input(Place):
        pass

    class BufferPlace(Place):
        MAX_CAPACITY = capacity

    class Output(Place):
        pass

    class BufferTransition(Transition):
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.INPUT: Arc(Input)}

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.BUFFER: Arc(BufferPlace)}

        async def on_fire(
            self, consumed: Dict[PlaceName, List]
        ) -> Dict[PlaceName, List]:
            return {BufferPlaceNames.BUFFER: consumed[BufferPlaceNames.INPUT]}

    class OutputTransition(Transition):
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.BUFFER: Arc(BufferPlace)}

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.OUTPUT: Arc(Output)}

        async def on_fire(
            self, consumed: Dict[PlaceName, List]
        ) -> Dict[PlaceName, List]:
            return {BufferPlaceNames.OUTPUT: consumed[BufferPlaceNames.BUFFER]}

    attrs.update(
        {
            "Input": Input,
            "BufferPlace": BufferPlace,
            "Output": Output,
            "BufferTransition": BufferTransition,
            "OutputTransition": OutputTransition,
            "_discovered_places": [Input, BufferPlace, Output],
            "_discovered_transitions": [BufferTransition, OutputTransition],
            "PlaceNames": BufferPlaceNames,
        }
    )

    return type("Buffer", (Interface,), attrs)


def RateLimiter(tokens_per_second: float) -> Type[Interface]:
    """
    Create a rate limiter interface that throttles token flow.

    Usage: class MyRateLimiter(RateLimiter(tokens_per_second=5.0)): pass
    """
    import time

    RateLimiterPlaceNames = Enum(
        "RateLimiterPlaceNames", {"INPUT": "input", "OUTPUT": "output"}, type=PlaceName
    )

    attrs = {}

    class Input(Place):
        pass

    class Output(Place):
        pass

    class RateLimiterTransition(Transition):
        def __init__(self):
            super().__init__()
            self._last_fire_time = 0
            self._min_interval = 1.0 / tokens_per_second

        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {RateLimiterPlaceNames.INPUT: Arc(Input)}

        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {RateLimiterPlaceNames.OUTPUT: Arc(Output)}

        async def on_before_fire(self, consumed: Dict[PlaceName, List]):
            # Implement rate limiting
            now = time.time()
            time_since_last = now - self._last_fire_time

            if time_since_last < self._min_interval:
                sleep_time = self._min_interval - time_since_last
                await sleep(sleep_time)

            self._last_fire_time = time.time()

        async def on_fire(
            self, consumed: Dict[PlaceName, List]
        ) -> Dict[PlaceName, List]:
            return {RateLimiterPlaceNames.OUTPUT: consumed[RateLimiterPlaceNames.INPUT]}

    attrs.update(
        {
            "Input": Input,
            "Output": Output,
            "RateLimiterTransition": RateLimiterTransition,
            "_discovered_places": [Input, Output],
            "_discovered_transitions": [RateLimiterTransition],
            "PlaceNames": RateLimiterPlaceNames,
        }
    )

    return type("RateLimiter", (Interface,), attrs)
