#!/usr/bin/env python3
"""
Cordyceps Asyncio Petri Net Utilities: Parametric Fork and Join Interfaces

Updated for the autonomous asyncio architecture with PlaceName enum support.
"""
from .core import Interface, Place, Transition, Arc, PlaceName
from typing import Type, Dict, List
from enum import Enum, auto
import asyncio
from asyncio import sleep


def ForkN(n: int) -> Type[Interface]:
    """
    Returns an Interface subclass with 1 input place, n output places, and a transition
    that replicates each input token to all outputs.
    
    Uses PlaceName enums for type safety.
    """
    # Create PlaceName enum for this fork
    place_names = ['INPUT'] + [f'OUTPUT_{i+1}' for i in range(n)]
    ForkPlaceNames = Enum(f'Fork{n}PlaceNames', {name: name.lower() for name in place_names}, type=PlaceName)
    
    attrs = {}
    
    # Create input place
    class Input(Place):
        pass
    attrs['Input'] = Input
    discovered_places = [Input]
    
    # Create output places
    output_classes = []
    for i in range(n):
        output_cls = type(f'Output_{i+1}', (Place,), {})
        attrs[f'Output_{i+1}'] = output_cls
        discovered_places.append(output_cls)
        output_classes.append(output_cls)
    
    # Create fork transition
    class ForkTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ALL  # Wait for input token
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {ForkPlaceNames.INPUT: Arc(Input)}
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            result = {}
            for i in range(n):
                enum_key = getattr(ForkPlaceNames, f'OUTPUT_{i+1}')
                result[enum_key] = Arc(output_classes[i])
            return result
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            # Replicate each input token to all outputs
            result = {}
            for i in range(n):
                enum_key = getattr(ForkPlaceNames, f'OUTPUT_{i+1}')
                result[enum_key] = []
            
            input_tokens = consumed[ForkPlaceNames.INPUT]
            for token in input_tokens:
                for i in range(n):
                    enum_key = getattr(ForkPlaceNames, f'OUTPUT_{i+1}')
                    result[enum_key].append(token)
            
            return result
    
    attrs['ForkTransition'] = ForkTransition
    discovered_transitions = [ForkTransition]
    
    attrs['_discovered_places'] = discovered_places
    attrs['_discovered_transitions'] = discovered_transitions
    attrs['PlaceNames'] = ForkPlaceNames  # Expose the enum
    
    clsname = f'Fork{n}'
    return type(clsname, (Interface,), attrs)


def JoinN(n: int) -> Type[Interface]:
    """
    Returns an Interface subclass with n input places, 1 output place, and a transition
    that moves all available tokens from inputs to the output.
    
    Uses PlaceName enums for type safety.
    """
    # Create PlaceName enum for this join
    place_names = [f'INPUT_{i+1}' for i in range(n)] + ['OUTPUT']
    JoinPlaceNames = Enum(f'Join{n}PlaceNames', {name: name.lower() for name in place_names}, type=PlaceName)
    
    attrs = {}
    discovered_places = []
    
    # Create input places
    input_classes = []
    for i in range(n):
        input_cls = type(f'Input_{i+1}', (Place,), {})
        attrs[f'Input_{i+1}'] = input_cls
        discovered_places.append(input_cls)
        input_classes.append(input_cls)
    
    # Create output place
    class Output(Place):
        pass
    attrs['Output'] = Output
    discovered_places.append(Output)
    
    # Create join transition
    class JoinTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ANY  # Fire when any input has tokens
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            result = {}
            for i in range(n):
                enum_key = getattr(JoinPlaceNames, f'INPUT_{i+1}')
                result[enum_key] = Arc(input_classes[i])
            return result
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {JoinPlaceNames.OUTPUT: Arc(Output)}
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            # Collect all tokens from all inputs
            all_tokens = []
            for tokens in consumed.values():
                all_tokens.extend(tokens)
            
            return {JoinPlaceNames.OUTPUT: all_tokens}
    
    attrs['JoinTransition'] = JoinTransition
    discovered_transitions = [JoinTransition]
    
    attrs['_discovered_places'] = discovered_places
    attrs['_discovered_transitions'] = discovered_transitions
    attrs['PlaceNames'] = JoinPlaceNames  # Expose the enum
    
    clsname = f'Join{n}'
    return type(clsname, (Interface,), attrs)


def Fork(from_place_class: Type[Place], to_place_classes: List[Type[Place]]) -> Type[Transition]:
    """
    Declaratively create a fork transition from one place to multiple places.
    Usage: Fork(MyInterface.InputPlace, [MyInterface.Output1, MyInterface.Output2])
    Returns a Transition subclass.
    """
    # Generate enum mapping automatically
    place_names = ['INPUT'] + [f'OUTPUT_{i+1}' for i in range(len(to_place_classes))]
    ForkPlaceNames = Enum('ForkPlaceNames', {name: name.lower() for name in place_names}, type=PlaceName)
    
    # Create mapping
    enum_mapping = {from_place_class: ForkPlaceNames.INPUT}
    for i, place_class in enumerate(to_place_classes):
        enum_key = getattr(ForkPlaceNames, f'OUTPUT_{i+1}')
        enum_mapping[place_class] = enum_key
    
    class _ForkTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ALL
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[from_place_class]: Arc(from_place_class)}
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[place_class]: Arc(place_class) for place_class in to_place_classes}
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            result = {enum_mapping[place_class]: [] for place_class in to_place_classes}
            
            input_tokens = consumed[enum_mapping[from_place_class]]
            for token in input_tokens:
                for place_class in to_place_classes:
                    result[enum_mapping[place_class]].append(token)
            
            return result
    
    return _ForkTransition


def Join(from_place_classes: List[Type[Place]], to_place_class: Type[Place]) -> Type[Transition]:
    """
    Declaratively create a join transition from multiple places to one place.
    Usage: Join([MyInterface.Input1, MyInterface.Input2], MyInterface.Output)
    Returns a Transition subclass.
    """
    # Generate enum mapping automatically
    place_names = [f'INPUT_{i+1}' for i in range(len(from_place_classes))] + ['OUTPUT']
    JoinPlaceNames = Enum('JoinPlaceNames', {name: name.lower() for name in place_names}, type=PlaceName)
    
    # Create mapping
    enum_mapping = {to_place_class: JoinPlaceNames.OUTPUT}
    for i, place_class in enumerate(from_place_classes):
        enum_key = getattr(JoinPlaceNames, f'INPUT_{i+1}')
        enum_mapping[place_class] = enum_key
    
    class _JoinTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ANY
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[place_class]: Arc(place_class) for place_class in from_place_classes}
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[to_place_class]: Arc(to_place_class)}
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            all_tokens = []
            for tokens in consumed.values():
                all_tokens.extend(tokens)
            
            return {enum_mapping[to_place_class]: all_tokens}
    
    return _JoinTransition


def Merge(place_classes: List[Type[Place]]) -> Type[Transition]:
    """
    Create a merge transition that forwards tokens from any of the input places
    to all of the same input places (creates a merge point).
    
    Usage: Merge([MyInterface.Place1, MyInterface.Place2])
    """
    # Generate enum mapping automatically
    place_names = [f'PLACE_{i+1}' for i in range(len(place_classes))]
    MergePlaceNames = Enum('MergePlaceNames', {name: name.lower() for name in place_names}, type=PlaceName)
    
    # Create mapping
    enum_mapping = {}
    for i, place_class in enumerate(place_classes):
        enum_key = getattr(MergePlaceNames, f'PLACE_{i+1}')
        enum_mapping[place_class] = enum_key
    
    class _MergeTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ANY
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[place_class]: Arc(place_class) for place_class in place_classes}
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {enum_mapping[place_class]: Arc(place_class) for place_class in place_classes}
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            # Forward all consumed tokens to all output places
            all_tokens = []
            for tokens in consumed.values():
                all_tokens.extend(tokens)
            
            result = {}
            for place_class in place_classes:
                result[enum_mapping[place_class]] = all_tokens.copy()
            
            return result
    
    return _MergeTransition


def Buffer(capacity: int) -> Type[Interface]:
    """
    Create a buffer interface with specified capacity.
    Useful for rate limiting and backpressure management.
    
    Usage: class MyBuffer(Buffer(capacity=10)): pass
    """
    BufferPlaceNames = Enum('BufferPlaceNames', {
        'INPUT': 'input',
        'BUFFER': 'buffer', 
        'OUTPUT': 'output'
    }, type=PlaceName)
    
    attrs = {}
    
    class Input(Place):
        pass
    
    class BufferPlace(Place):
        MAX_CAPACITY = capacity
    
    class Output(Place):
        pass
    
    class BufferTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ALL
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.INPUT: Arc(Input)}
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.BUFFER: Arc(BufferPlace)}
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            return {BufferPlaceNames.BUFFER: consumed[BufferPlaceNames.INPUT]}
    
    class OutputTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ALL
        
        def input_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.BUFFER: Arc(BufferPlace)}
        
        def output_arcs(self) -> Dict[PlaceName, Arc]:
            return {BufferPlaceNames.OUTPUT: Arc(Output)}
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            return {BufferPlaceNames.OUTPUT: consumed[BufferPlaceNames.BUFFER]}
    
    attrs.update({
        'Input': Input,
        'BufferPlace': BufferPlace,
        'Output': Output,
        'BufferTransition': BufferTransition,
        'OutputTransition': OutputTransition,
        '_discovered_places': [Input, BufferPlace, Output],
        '_discovered_transitions': [BufferTransition, OutputTransition],
        'PlaceNames': BufferPlaceNames
    })
    
    return type('Buffer', (Interface,), attrs)


def RateLimiter(tokens_per_second: float) -> Type[Interface]:
    """
    Create a rate limiter interface that throttles token flow.
    
    Usage: class MyRateLimiter(RateLimiter(tokens_per_second=5.0)): pass
    """
    import time
    
    RateLimiterPlaceNames = Enum('RateLimiterPlaceNames', {
        'INPUT': 'input',
        'OUTPUT': 'output'
    }, type=PlaceName)
    
    attrs = {}
    
    class Input(Place):
        pass
    
    class Output(Place):
        pass
    
    class RateLimiterTransition(Transition):
        DISPATCH_POLICY = DispatchPolicy.ALL
        
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
        
        async def on_fire(self, consumed: Dict[PlaceName, List]) -> Dict[PlaceName, List]:
            return {RateLimiterPlaceNames.OUTPUT: consumed[RateLimiterPlaceNames.INPUT]}
    
    attrs.update({
        'Input': Input,
        'Output': Output,
        'RateLimiterTransition': RateLimiterTransition,
        '_discovered_places': [Input, Output],
        '_discovered_transitions': [RateLimiterTransition],
        'PlaceNames': RateLimiterPlaceNames
    })
    
    return type('RateLimiter', (Interface,), attrs)