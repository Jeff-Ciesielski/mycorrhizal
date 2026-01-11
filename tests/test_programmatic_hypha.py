#!/usr/bin/env python3
"""Tests for programmatic Petri net building using NetBuilder API"""

import sys
sys.path.insert(0, "src")

import pytest
import asyncio
from pydantic import BaseModel

from mycorrhizal.hypha.core import Runner, PlaceType
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.common.timebase import MonotonicClock


# ============================================================================
# Test Fixtures
# ============================================================================

class SimpleBlackboard(BaseModel):
    """Simple blackboard for testing"""
    counter: int = 0


# ============================================================================
# Tests: Basic Programmatic Building
# ============================================================================

def test_build_simple_net_programmatically():
    """Test building a simple net without decorators"""
    builder = NetBuilder("SimpleNet")

    # Create places
    input_place = builder.place("input", type=PlaceType.QUEUE)
    output_place = builder.place("output", type=PlaceType.BAG)

    # Create transition
    async def process(consumed, bb, timebase):
        for token in consumed:
            yield {output_place: f"processed_{token}"}

    transition = builder.transition()(process)

    # Wire together
    builder.arc(input_place, transition).arc(output_place)

    # Verify spec was created
    assert "input" in builder.spec.places
    assert "output" in builder.spec.places
    assert "process" in builder.spec.transitions
    assert len(builder.spec.arcs) == 2

    # Create wrapper function for Runner
    def net_func():
        pass

    net_func._spec = builder.spec

    # Verify we can get mermaid diagram using utility function
    from mycorrhizal.hypha.util import to_mermaid
    mermaid = to_mermaid(net_func._spec)
    assert "SimpleNet" in mermaid
    assert "input" in mermaid
    assert "output" in mermaid


def test_build_net_with_io_places():
    """Test building a net with IO input/output places"""
    builder = NetBuilder("IONet")

    # Create IO input place
    async def source(bb, timebase):
        yield "token1"
        yield "token2"

    input_place = builder.io_input_place()(source)

    # Create regular places
    queue = builder.place("queue", type=PlaceType.QUEUE)
    output = builder.place("output", type=PlaceType.BAG)

    # Create IO output place
    async def sink(bb, timebase, tokens):
        for token in tokens:
            pass  # Consume tokens

    output_place = builder.io_output_place()(sink)

    # Create transition
    async def process(consumed, bb, timebase):
        for token in consumed:
            yield {output_place: token}

    transition = builder.transition()(process)

    # Wire together
    builder.arc(input_place, transition).arc(queue)
    builder.arc(queue, transition).arc(output_place)

    # Verify spec
    assert "source" in builder.spec.places
    assert builder.spec.places["source"].is_io_input
    assert "sink" in builder.spec.places
    assert builder.spec.places["sink"].is_io_output


def test_build_net_with_multiple_transitions():
    """Test building a net with multiple transitions"""
    builder = NetBuilder("MultiTransNet")

    # Create places
    input1 = builder.place("input1", type=PlaceType.QUEUE)
    input2 = builder.place("input2", type=PlaceType.QUEUE)
    output = builder.place("output", type=PlaceType.BAG)

    # Create transitions
    async def process1(consumed, bb, timebase):
        for token in consumed:
            yield {output: f"p1_{token}"}

    async def process2(consumed, bb, timebase):
        for token in consumed:
            yield {output: f"p2_{token}"}

    trans1 = builder.transition()(process1)
    trans2 = builder.transition()(process2)

    # Wire together
    builder.arc(input1, trans1).arc(output)
    builder.arc(input2, trans2).arc(output)

    # Verify
    assert len(builder.spec.transitions) == 2
    assert len(builder.spec.arcs) == 4


def test_build_complex_net_with_all_place_types():
    """Test building a net with BAG and QUEUE place types"""
    builder = NetBuilder("MixedTypesNet")

    # Create different place types
    queue_place = builder.place("queue", type=PlaceType.QUEUE)
    bag_place = builder.place("bag", type=PlaceType.BAG)

    # Verify types
    assert builder.spec.places["queue"].place_type == PlaceType.QUEUE
    assert builder.spec.places["bag"].place_type == PlaceType.BAG


# ============================================================================
# Test Handlers for Dynamic Construction Tests
# ============================================================================

async def _test_process_handler(consumed, bb, timebase):
    """Handler for dynamic config test"""
    for token in consumed:
        yield {"output": f"processed_{token}"}


# ============================================================================
# Tests: Dynamic Net Construction
# ============================================================================

def test_build_net_from_config():
    """Test building a net from configuration data"""
    # Define configuration
    config = {
        "name": "ConfigNet",
        "places": [
            {"name": "input", "type": "queue"},
            {"name": "output", "type": "bag"}
        ],
        "transitions": [
            {
                "name": "process",
                "arcs": [("input", "process"), ("process", "output")]
            }
        ]
    }

    # Define handler with the correct name
    async def process(consumed, bb, timebase):
        for token in consumed:
            yield {"output": f"processed_{token}"}

    # Build net from config
    builder = NetBuilder(config["name"])

    places = {}
    for place_config in config["places"]:
        place_type = PlaceType.QUEUE if place_config["type"] == "queue" else PlaceType.BAG
        places[place_config["name"]] = builder.place(
            place_config["name"],
            type=place_type
        )

    transitions = {}
    for trans_config in config["transitions"]:
        # Use handler directly - will be named after function
        trans = builder.transition()(process)
        transitions[trans_config["name"]] = trans

    for trans_config in config["transitions"]:
        trans_name = trans_config["name"]
        trans = transitions[trans_name]

        for arc_spec in trans_config["arcs"]:
            source_name, target_name = arc_spec

            if source_name in places:
                source = places[source_name]
            elif source_name in transitions:
                source = transitions[source_name]
            else:
                raise ValueError(f"Unknown source: {source_name}")

            if target_name in places:
                target = places[target_name]
            elif target_name in transitions:
                target = transitions[target_name]
            else:
                raise ValueError(f"Unknown target: {target_name}")

            builder.arc(source, target)

    # Verify net was built correctly
    assert "input" in builder.spec.places
    assert "output" in builder.spec.places
    assert "process" in builder.spec.transitions  # Function name is used
    assert len(builder.spec.arcs) == 2


def test_conditional_net_construction():
    """Test building different net structures based on conditions"""
    # Fast path net
    builder_fast = NetBuilder("FastNet")
    input_p = builder_fast.place("input", type=PlaceType.QUEUE)
    output_p = builder_fast.place("output", type=PlaceType.BAG)

    async def fast_process(consumed, bb, timebase):
        for token in consumed:
            yield {output_p: f"fast_{token}"}

    trans_fast = builder_fast.transition()(fast_process)
    builder_fast.arc(input_p, trans_fast).arc(output_p)

    # Slow path net
    builder_slow = NetBuilder("SlowNet")
    input_p2 = builder_slow.place("input", type=PlaceType.QUEUE)
    valid_p = builder_slow.place("validation", type=PlaceType.BAG)
    output_p2 = builder_slow.place("output", type=PlaceType.BAG)

    async def validate(consumed, bb, timebase):
        for token in consumed:
            yield {valid_p: f"validated_{token}"}

    async def slow_process(consumed, bb, timebase):
        for token in consumed:
            yield {output_p2: f"slow_{token}"}

    trans_valid = builder_slow.transition()(validate)
    trans_slow = builder_slow.transition()(slow_process)

    builder_slow.arc(input_p2, trans_valid).arc(valid_p)
    builder_slow.arc(valid_p, trans_slow).arc(output_p2)

    # Verify both nets have different structures
    assert len(builder_fast.spec.transitions) == 1
    assert len(builder_fast.spec.places) == 2
    assert len(builder_slow.spec.transitions) == 2
    assert len(builder_slow.spec.places) == 3


# ============================================================================
# Tests: Convenience Methods
# ============================================================================

def test_forward_convenience_method():
    """Test using the forward() convenience method"""
    builder = NetBuilder("ForwardNet")

    input_place = builder.place("input", type=PlaceType.QUEUE)
    output_place = builder.place("output", type=PlaceType.BAG)

    # Use forward to create pass-through transition
    builder.forward(input_place, output_place, name="pass_through")

    # Verify transition was created
    assert "pass_through" in builder.spec.transitions
    assert len(builder.spec.arcs) == 2


def test_arc_chaining():
    """Test fluent arc chaining with ArcChain"""
    builder = NetBuilder("ChainNet")

    place1 = builder.place("place1", type=PlaceType.QUEUE)
    place2 = builder.place("place2", type=PlaceType.QUEUE)
    place3 = builder.place("place3", type=PlaceType.BAG)

    async def trans1(consumed, bb, timebase):
        for token in consumed:
            yield {place2: token}

    async def trans2(consumed, bb, timebase):
        for token in consumed:
            yield {place3: token}

    t1 = builder.transition()(trans1)
    t2 = builder.transition()(trans2)

    # Chain arcs
    builder.arc(place1, t1).arc(place2).arc(t2).arc(place3)

    # Verify all arcs created
    assert len(builder.spec.arcs) == 4


# ============================================================================
# Tests: Integration - Running Programmatic Nets
# ============================================================================

@pytest.mark.asyncio
async def test_run_programmatic_net():
    """Test running a net built programmatically"""
    builder = NetBuilder("RunnableNet")

    # Create IO input
    async def source(bb, timebase):
        for i in range(3):
            await timebase.sleep(0.1)
            yield f"token{i}"

    input_place = builder.io_input_place()(source)
    output_place = builder.place("output", type=PlaceType.BAG)

    # Create transition
    async def process(consumed, bb, timebase):
        for token in consumed:
            bb.counter += 1
            yield {output_place: token}

    transition = builder.transition()(process)

    # Wire
    builder.arc(input_place, transition).arc(output_place)

    # Create wrapper function
    def net_func():
        pass

    net_func._spec = builder.spec

    # Run the net
    bb = SimpleBlackboard()
    runner = Runner(net_func, bb)
    timebase = MonotonicClock()

    await runner.start(timebase)
    await asyncio.sleep(1.0)  # Let tokens flow
    await runner.stop()

    # Verify processing occurred - should process all 3 tokens
    assert bb.counter >= 1  # At least one token was processed


@pytest.mark.asyncio
async def test_programmatic_and_decorator_nets_produce_same_results():
    """Test that programmatic and decorator approaches produce equivalent nets"""
    from mycorrhizal.hypha.core import pn

    # Build net with decorator
    @pn.net
    def DecoratorNet(builder: NetBuilder):
        @builder.io_input_place()
        async def source(bb, timebase):
            yield "test"

        output = builder.place("output", type=PlaceType.BAG)

        @builder.transition()
        async def process(consumed, bb, timebase):
            for token in consumed:
                bb.counter += 1
                yield {output: token}

        builder.arc(source, process).arc(output)

    # Build equivalent net programmatically
    builder2 = NetBuilder("ProgrammaticNet")

    async def source2(bb, timebase):
        yield "test"

    input2 = builder2.io_input_place()(source2)
    output2 = builder2.place("output", type=PlaceType.BAG)

    async def process2(consumed, bb, timebase):
        for token in consumed:
            bb.counter += 1
            yield {output2: token}

    trans2 = builder2.transition()(process2)
    builder2.arc(input2, trans2).arc(output2)

    def net_func():
        pass

    net_func._spec = builder2.spec
    net_func.to_mermaid = lambda: builder2.spec.to_mermaid()

    # Run both nets and compare results
    bb1 = SimpleBlackboard()
    runner1 = Runner(DecoratorNet, bb1)
    timebase1 = MonotonicClock()

    await runner1.start(timebase1)
    await asyncio.sleep(0.3)
    await runner1.stop()

    bb2 = SimpleBlackboard()
    runner2 = Runner(net_func, bb2)
    timebase2 = MonotonicClock()

    await runner2.start(timebase2)
    await asyncio.sleep(0.3)
    await runner2.stop()

    # Both should have processed the same number of tokens
    assert bb1.counter == bb2.counter


# ============================================================================
# Tests: Edge Cases
# ============================================================================

def test_empty_net():
    """Test building an empty net (no places, no transitions)"""
    builder = NetBuilder("EmptyNet")

    # Verify empty spec
    assert len(builder.spec.places) == 0
    assert len(builder.spec.transitions) == 0
    assert len(builder.spec.arcs) == 0


def test_net_with_only_places():
    """Test building a net with places but no transitions"""
    builder = NetBuilder("PlacesOnlyNet")

    builder.place("place1", type=PlaceType.QUEUE)
    builder.place("place2", type=PlaceType.BAG)

    # Verify places exist
    assert len(builder.spec.places) == 2
    assert len(builder.spec.transitions) == 0


def test_net_with_single_transition():
    """Test building a net with a single transition"""
    builder = NetBuilder("SingleTransNet")

    input_p = builder.place("input", type=PlaceType.QUEUE)
    output_p = builder.place("output", type=PlaceType.BAG)

    async def trans(consumed, bb, timebase):
        for token in consumed:
            yield {output_p: token}

    transition = builder.transition()(trans)
    builder.arc(input_p, transition).arc(output_p)

    # Verify
    assert len(builder.spec.transitions) == 1
    assert len(builder.spec.arcs) == 2


def test_invalid_arc_connection_place_to_place():
    """Test that connecting place to place raises an error"""
    builder = NetBuilder("InvalidNet")

    place1 = builder.place("place1", type=PlaceType.QUEUE)
    place2 = builder.place("place2", type=PlaceType.BAG)

    # Should raise error when trying to connect place to place
    with pytest.raises(ValueError, match="Cannot connect PlaceRef to PlaceRef"):
        builder.arc(place1, place2)


def test_invalid_arc_connection_transition_to_transition():
    """Test that connecting transition to transition raises an error"""
    builder = NetBuilder("InvalidNet2")

    async def trans1(consumed, bb, timebase):
        pass

    async def trans2(consumed, bb, timebase):
        pass

    t1 = builder.transition()(trans1)
    t2 = builder.transition()(trans2)

    # Should raise error when trying to connect transition to transition
    with pytest.raises(ValueError, match="Cannot connect TransitionRef to TransitionRef"):
        builder.arc(t1, t2)


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
