#!/usr/bin/env python3
"""
Hypha - Decorator-based Colored Petri Net Framework

A DSL for defining and executing colored Petri nets with support for asyncio,
various place types, guards, and hierarchical subnet composition.

Usage:
    from mycorrhizal.hypha.core import pn, Runner, PlaceType

    @pn.net
    def ProcessingNet(builder):
        # Define places
        pending = builder.place("pending", type=PlaceType.QUEUE)
        processed = builder.place("processed", type=PlaceType.QUEUE)

        # Define transitions
        @builder.transition()
        async def process(consumed, bb, timebase):
            for token in consumed:
                result = await handle(token)
                yield {processed: result}

        # Wire the net
        builder.arc(pending, process).arc(processed)

    # Run the Petri net
    runner = Runner(ProcessingNet, bb=blackboard)
    await runner.start(timebase)

Key Classes:
    NetBuilder - Builder for constructing Petri nets
    PlaceType - Type of place (QUEUE, SET, FLAG, COUNTER)
    PlaceSpec - Specification for a place
    TransitionSpec - Specification for a transition
    Runner - Runtime for executing Petri nets

Place Types:
    QUEUE - FIFO queue, allows multiple tokens
    SET - Unordered collection with unique tokens
    FLAG - Boolean place (has token or not)
    COUNTER - Numeric place with bounded capacity

Transitions:
    Transitions consume tokens from input places and produce tokens for output places.
    They are async functions that yield dictionaries mapping place references to tokens.

Guards:
    Guards are conditions that must be satisfied for a transition to fire.
    They can check token values, blackboard state, or external conditions.
"""

from .specs import (
    PlaceType,
    PlaceSpec,
    GuardSpec,
    TransitionSpec,
    ArcSpec,
    NetSpec,
    PlaceRef,
    TransitionRef,
    SubnetRef,
)

from .builder import (
    NetBuilder,
    ArcChain,
    PetriNetDSL,
    pn,
)

from .runtime import (
    PlaceRuntime,
    TransitionRuntime,
    NetRuntime,
    Runner,
)

__all__ = [
    # Core types
    'PlaceType',
    'GuardSpec',
    
    # Specification types (for type hints)
    'PlaceSpec',
    'TransitionSpec',
    'ArcSpec',
    'NetSpec',
    
    # Reference types
    'PlaceRef',
    'TransitionRef',
    'SubnetRef',
    
    # Builder
    'NetBuilder',
    'ArcChain',
    
    # Main API
    'pn',
    
    # Runtime
    'PlaceRuntime',
    'TransitionRuntime',
    'NetRuntime',
    'Runner',
]