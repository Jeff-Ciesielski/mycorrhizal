#!/usr/bin/env python3
"""
Programmatic Hypha Net Building Demo

This example demonstrates how to build Petri nets programmatically using the
NetBuilder API, without using the @pn.net decorator DSL.

When to use programmatic building:
- Dynamic net construction based on configuration/data
- Runtime workflow generation
- Building nets from external configs (JSON, YAML, etc.)
- Alternative to decorator syntax for programmatic use cases
"""

import asyncio
from pydantic import BaseModel

from mycorrhizal.hypha.core import Runner, PlaceType
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.common.timebase import MonotonicClock


# ============================================================================
# Example 1: Simple Linear Workflow (Programmatic vs Decorator)
# ============================================================================

class SimpleBlackboard(BaseModel):
    """Simple blackboard for demonstration"""
    counter: int = 0


# ----------------------------------------------------------------------
# APPROACH 1: Using @pn.net decorator (traditional)
# ----------------------------------------------------------------------
from mycorrhizal.hypha.core import pn

@pn.net
def DecoratorNet(builder: NetBuilder):
    """Traditional decorator-based net definition"""

    @builder.io_input_place()
    async def source(bb, timebase):
        """Token source"""
        for i in range(3):
            await timebase.sleep(0.1)
            yield f"token_{i}"

    queue = builder.place("queue", type=PlaceType.QUEUE)
    processed = builder.place("processed", type=PlaceType.BAG)

    @builder.transition()
    async def process(consumed, bb, timebase):
        """Process tokens"""
        for token in consumed:
            print(f"  [Decorator] Processing: {token}")
            bb.counter += 1
            yield {processed: token}

    # Wire it together
    builder.arc(source, process).arc(queue)
    builder.arc(queue, process).arc(processed)


# ----------------------------------------------------------------------
# APPROACH 2: Using NetBuilder API directly (programmatic)
# ----------------------------------------------------------------------
def build_programmatic_net():
    """Programmatic net construction without decorators"""

    # Create a new builder
    builder = NetBuilder("ProgrammaticNet")

    # Define IO input place
    async def source_func(bb, timebase):
        """Token source"""
        for i in range(3):
            await timebase.sleep(0.1)
            yield f"token_{i}"

    # Register the IO input place
    source = builder.io_input_place()(source_func)

    # Define regular places
    queue = builder.place("queue", type=PlaceType.QUEUE)
    processed = builder.place("processed", type=PlaceType.BAG)

    # Define transition function
    async def process_func(consumed, bb, timebase):
        """Process tokens"""
        for token in consumed:
            print(f"  [Programmatic] Processing: {token}")
            bb.counter += 1
            yield {processed: token}

    # Register the transition
    process = builder.transition()(process_func)

    # Wire it together
    builder.arc(source, process).arc(queue)
    builder.arc(queue, process).arc(processed)

    # Create a wrapper function to hold the spec (for Runner compatibility)
    def programmatic_net_func():
        pass

    programmatic_net_func._spec = builder.spec

    return programmatic_net_func


# ============================================================================
# Example 2: Dynamic Net Construction from Configuration
# ============================================================================

def build_net_from_config(config: dict):
    """
    Build a Petri net dynamically from configuration data.

    This demonstrates the key advantage of programmatic building:
    constructing nets at runtime based on external data.
    """

    # Config structure:
    # {
    #     "name": "MyNet",
    #     "places": [
    #         {"name": "input", "type": "queue"},
    #         {"name": "output", "type": "bag"}
    #     ],
    #     "transitions": [
    #         {
    #             "name": "process",
    #             "handler": "my_process_func",
    #             "arcs": [("input", "process"), ("process", "output")]
    #         }
    #     ]
    # }

    builder = NetBuilder(config["name"])

    # Create places from config
    places = {}
    for place_config in config["places"]:
        place_type = PlaceType.QUEUE if place_config["type"] == "queue" else PlaceType.BAG
        places[place_config["name"]] = builder.place(
            place_config["name"],
            type=place_type
        )

    # Create transitions from config
    transitions = {}
    for trans_config in config["transitions"]:
        # Get handler function from globals (in real use, register handlers)
        handler = globals()[trans_config["handler"]]
        trans = builder.transition()(handler)
        transitions[trans_config["name"]] = trans

    # Create arcs from config
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

    # Create wrapper function for Runner compatibility
    def net_func():
        pass

    net_func._spec = builder.spec

    return net_func


# Handler for dynamic net
async def dynamic_process(consumed, bb, timebase):
    """Process handler for dynamically built net"""
    for token in consumed:
        print(f"  [Dynamic] Processing: {token}")
        yield {"output": f"processed_{token}"}


# ============================================================================
# Example 3: Conditional Net Construction
# ============================================================================

def build_conditional_net(use_fast_path: bool):
    """
    Build different net structures based on runtime conditions.

    This demonstrates conditional workflow construction.
    """

    builder = NetBuilder("ConditionalNet")

    # Common places
    input_place = builder.place("input", type=PlaceType.QUEUE)
    output_place = builder.place("output", type=PlaceType.BAG)

    if use_fast_path:
        # Fast path: direct processing
        @builder.transition()
        async def fast_process(consumed, bb, timebase):
            for token in consumed:
                print(f"  [Fast Path] Processing: {token}")
                yield {output_place: f"fast_{token}"}

        builder.arc(input_place, fast_process).arc(output_place)

    else:
        # Slow path: validation then processing
        validation = builder.place("validation", type=PlaceType.BAG)

        @builder.transition()
        async def validate(consumed, bb, timebase):
            for token in consumed:
                print(f"  [Slow Path] Validating: {token}")
                yield {validation: f"validated_{token}"}

        @builder.transition()
        async def slow_process(consumed, bb, timebase):
            for token in consumed:
                print(f"  [Slow Path] Processing: {token}")
                yield {output_place: f"slow_{token}"}

        builder.arc(input_place, validate).arc(validation)
        builder.arc(validation, slow_process).arc(output_place)

    # Create wrapper function
    def net_func():
        pass

    net_func._spec = builder.spec

    return net_func


# ============================================================================
# Example 4: Convenience Methods (forward, fork, join, etc.)
# ============================================================================

def build_net_with_convenience_methods():
    """
    Demonstrate NetBuilder convenience methods for common patterns.
    """

    builder = NetBuilder("ConvenienceNet")

    # Create places
    input1 = builder.place("input1", type=PlaceType.QUEUE)
    input2 = builder.place("input2", type=PlaceType.QUEUE)
    output = builder.place("output", type=PlaceType.BAG)

    # Use forward() for simple pass-through
    builder.forward(input1, output, name="forward1")
    builder.forward(input2, output, name="forward2")

    # Create wrapper function
    def net_func():
        pass

    net_func._spec = builder.spec

    return net_func


# ============================================================================
# Demonstration
# ============================================================================

async def demo_decorator_net():
    """Demo the decorator-based approach"""
    print("\n" + "="*70)
    print("Decorator-Based Net")
    print("="*70)

    bb = SimpleBlackboard()
    runner = Runner(DecoratorNet, bb)
    timebase = MonotonicClock()

    from mycorrhizal.hypha.util import to_mermaid

    print("\nMermaid Diagram:")
    print(to_mermaid(DecoratorNet._spec))

    await runner.start(timebase)
    await asyncio.sleep(1)
    await runner.stop()

    print(f"\nFinal counter: {bb.counter}")


async def demo_programmatic_net():
    """Demo the programmatic approach"""
    print("\n" + "="*70)
    print("Programmatic Net")
    print("="*70)

    # Build the net programmatically
    net_spec = build_programmatic_net()

    bb = SimpleBlackboard()
    runner = Runner(net_spec, bb)
    timebase = MonotonicClock()

    from mycorrhizal.hypha.util import to_mermaid

    print("\nMermaid Diagram:")
    print(to_mermaid(net_spec._spec))

    await runner.start(timebase)
    await asyncio.sleep(1)
    await runner.stop()

    print(f"\nFinal counter: {bb.counter}")


async def demo_dynamic_net():
    """Demo dynamic net construction from config"""
    print("\n" + "="*70)
    print("Dynamic Net Construction")
    print("="*70)

    # Define configuration
    config = {
        "name": "DynamicNet",
        "places": [
            {"name": "input", "type": "queue"},
            {"name": "output", "type": "bag"}
        ],
        "transitions": [
            {
                "name": "process",
                "handler": "dynamic_process",
                "arcs": [("input", "process"), ("process", "output")]
            }
        ]
    }

    # Build net from config
    net_spec = build_net_from_config(config)

    class DynamicBlackboard(BaseModel):
        pass

    bb = DynamicBlackboard()
    runner = Runner(net_spec, bb)
    timebase = MonotonicClock()

    from mycorrhizal.hypha.util import to_mermaid

    print("\nMermaid Diagram:")
    print(to_mermaid(net_spec._spec))

    # Add tokens to input place
    await runner.start(timebase)

    runtime = runner.runtime
    input_queue = runtime.places[("DynamicNet", "input")]
    for i in range(3):
        input_queue.add_token(f"data_{i}")

    await asyncio.sleep(1)
    await runner.stop()

    # Check results
    output_bag = runtime.places[("DynamicNet", "output")]
    print(f"\nOutput tokens: {list(output_bag.tokens)}")


async def demo_conditional_net():
    """Demo conditional net construction"""
    print("\n" + "="*70)
    print("Conditional Net Construction")
    print("="*70)

    print("\n--- Building fast path net ---")
    fast_net = build_conditional_net(use_fast_path=True)
    from mycorrhizal.hypha.util import to_mermaid

    class ConditionalBlackboard(BaseModel):
        pass

    bb = ConditionalBlackboard()
    runner = Runner(fast_net, bb)
    print(to_mermaid(fast_net._spec))

    print("\n--- Building slow path net ---")
    slow_net = build_conditional_net(use_fast_path=False)
    print(to_mermaid(slow_net._spec))


async def demo_convenience_methods():
    """Demo convenience methods"""
    print("\n" + "="*70)
    print("Convenience Methods")
    print("="*70)

    net_spec = build_net_with_convenience_methods()
    from mycorrhizal.hypha.util import to_mermaid

    print("\nMermaid Diagram:")
    print(to_mermaid(net_spec._spec))


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demonstrations"""
    print("="*70)
    print("Programmatic Hypha Net Building Demo")
    print("="*70)

    await demo_decorator_net()
    await demo_programmatic_net()
    await demo_dynamic_net()
    await demo_conditional_net()
    await demo_convenience_methods()

    print("\n" + "="*70)
    print("Key Takeaways")
    print("="*70)
    print("""
1. Decorator DSL (@pn.net):
   - Clean, declarative syntax
   - Best for static net structures
   - Easier to read and maintain

2. Programmatic API (NetBuilder):
   - Build nets at runtime
   - Dynamic structures from config/data
   - Conditional workflow paths
   - Alternative to decorator syntax

3. When to use programmatic:
   - Configuration-driven workflows
   - Runtime net generation
   - External workflow definitions
   - Dynamic topology changes

4. NetBuilder API methods:
   - place() - Create regular places
   - io_input_place() - Create IO input (decorator)
   - io_output_place() - Create IO output (decorator)
   - transition() - Create transitions (decorator)
   - arc() - Connect places/transitions
   - forward() - Simple pass-through
   - fork() - Split to multiple outputs
   - join() - Merge from multiple inputs
   - subnet() - Instantiate subnets
    """)


if __name__ == "__main__":
    asyncio.run(main())
