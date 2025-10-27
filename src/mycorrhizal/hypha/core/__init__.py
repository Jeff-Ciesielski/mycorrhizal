#!/usr/bin/env python3
"""
mycorrhizal.hypha.core - Decorator-based Petri Net DSL

Public API for defining and executing Petri nets using a decorator-based syntax.
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