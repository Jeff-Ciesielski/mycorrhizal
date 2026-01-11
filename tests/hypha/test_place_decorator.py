#!/usr/bin/env python3
"""Tests for place decorator syntax variations"""

import sys
sys.path.insert(0, "src")

from mycorrhizal.hypha.core import pn, PlaceType, NetBuilder
from pydantic import BaseModel
from typing import List


class TestBB(BaseModel):
    tokens: List[str] = []
    processed: List[str] = []


# Test @builder.place without parens (infers name)
@pn.net
def PlaceNoParens(builder: NetBuilder):

    @builder.place
    def input_queue(bb: TestBB):
        return bb.tokens

    @builder.transition()
    async def process(consumed, bb, timebase):
        token = consumed[0].payload
        bb.processed.append(token)
        return []

    builder.arc(input_queue, process)


# Test @builder.place("custom_name") with custom name
@pn.net
def PlaceWithCustomName(builder: NetBuilder):

    my_queue = builder.place("my_queue")

    @my_queue
    def input_func(bb: TestBB):
        return bb.tokens

    @builder.transition()
    async def process(consumed, bb, timebase):
        token = consumed[0].payload
        bb.processed.append(token)
        return []

    builder.arc(my_queue, process)


# Test traditional method call style
@pn.net
def PlaceMethodCall(builder: NetBuilder):

    input_queue = builder.place("input_queue", type=PlaceType.QUEUE)

    @builder.transition()
    async def process(consumed, bb, timebase):
        token = consumed[0].payload
        bb.processed.append(token)
        return []

    builder.arc(input_queue, process)


# Tests

def test_place_no_parens_creates_place():
    """Test @builder.place without parentheses creates a place"""
    spec = PlaceNoParens._spec
    assert "input_queue" in spec.places
    assert spec.places["input_queue"].place_type == PlaceType.BAG
    assert spec.places["input_queue"].handler is not None


def test_place_no_parens_infers_name():
    """Test @builder.place infers name from function name"""
    spec = PlaceNoParens._spec
    # The place name should match the function name
    assert "input_queue" in spec.places


def test_place_with_custom_name():
    """Test @builder.place('custom_name') uses provided name"""
    spec = PlaceWithCustomName._spec
    # The place name should be the custom name, not function name
    assert "my_queue" in spec.places
    assert "input_func" not in spec.places


def test_place_method_call():
    """Test traditional method call style still works"""
    spec = PlaceMethodCall._spec
    assert "input_queue" in spec.places
    assert spec.places["input_queue"].place_type == PlaceType.QUEUE


def test_place_handler_registered():
    """Test that handler function is registered when using decorator"""
    spec = PlaceNoParens._spec
    place_spec = spec.places["input_queue"]
    assert place_spec.handler is not None
    assert callable(place_spec.handler)


def test_place_no_handler_when_method_call():
    """Test that no handler is registered when using method call"""
    spec = PlaceMethodCall._spec
    place_spec = spec.places["input_queue"]
    # Method call style doesn't have a handler
    assert place_spec.handler is None


def test_place_with_type_and_decorator():
    """Test @builder.place with type parameter and handler"""
    @pn.net
    def TypedPlace(builder: NetBuilder):

        my_queue = builder.place(type=PlaceType.QUEUE)

        @my_queue
        def queue_handler(bb: TestBB):
            return bb.tokens

    spec = TypedPlace._spec
    assert "queue_handler" in spec.places
    assert spec.places["queue_handler"].place_type == PlaceType.QUEUE
    assert spec.places["queue_handler"].handler is not None


def test_place_decorator_returns_place_ref():
    """Test that place decorator returns PlaceRef for use in arcs"""
    @pn.net
    def RefTest(builder: NetBuilder):

        my_place = builder.place("my_place")

        @builder.transition()
        async def dummy(consumed, bb, timebase):
            return []

        # Should be able to use in arc
        builder.arc(my_place, dummy)

    spec = RefTest._spec
    assert "my_place" in spec.places


def test_multiple_place_decorators():
    """Test multiple places using decorator syntax"""
    @pn.net
    def MultiplePlaces(builder: NetBuilder):

        place1 = builder.place()

        @place1
        def place1_func(bb: TestBB):
            return bb.tokens

        place2 = builder.place()

        @place2
        def place2_func(bb: TestBB):
            return bb.processed

    spec = MultiplePlaces._spec
    assert "place1_func" in spec.places
    assert "place2_func" in spec.places
    assert spec.places["place1_func"].handler is not None
    assert spec.places["place2_func"].handler is not None
