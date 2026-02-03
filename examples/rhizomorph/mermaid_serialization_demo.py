#!/usr/bin/env python3
"""
Example: Mermaid Export and Serialization for TreeBuilder

Demonstrates:
1. Generating Mermaid diagrams from programmatic trees
2. Serializing trees to JSON
3. Using these for documentation and debugging
"""

import asyncio
import json
from dataclasses import dataclass


from mycorrhizal.rhizomorph import TreeBuilder, Runner, Status


@dataclass
class SimpleBlackboard:
    """Simple blackboard for demos"""
    value: int = 0


# ============================================================================
# Demo 1: Mermaid Export
# ============================================================================


async def demo_mermaid_export():
    """Generate Mermaid diagram from a programmatic tree."""
    print("=" * 60)
    print("Demo 1: Mermaid Diagram Export")
    print("=" * 60)

    builder = TreeBuilder("PatrolTree")

    async def check_battery(bb):
        return bb.value > 20

    async def scan(bb):
        bb.value += 1
        return Status.SUCCESS

    async def move(bb):
        bb.value += 1
        return Status.SUCCESS

    battery_ok = builder.condition("battery_ok", check_battery)
    scan_action = builder.action("scan", scan)
    move_action = builder.action("move", move)

    patrol = builder.sequence(battery_ok, move_action, memory=True)
    root = builder.selector(scan_action, patrol, memory=False)

    tree = builder.build(root)

    # Generate Mermaid diagram
    mermaid = tree.to_mermaid()

    print("\n--- Mermaid Diagram ---")
    print(mermaid)
    print("--- End of Diagram ---\n")

    print("Render this diagram at: https://mermaid.live/")
    print()


# ============================================================================
# Demo 2: JSON Serialization
# ============================================================================


async def demo_serialization():
    """Serialize tree to JSON for saving/sharing."""
    print("\n" + "=" * 60)
    print("Demo 2: JSON Serialization")
    print("=" * 60)

    builder = TreeBuilder("SaveableTree")

    async def action_a(bb):
        return Status.SUCCESS

    async def action_b(bb):
        return Status.FAILURE

    async def action_c(bb):
        return Status.SUCCESS

    a = builder.action("action_a", action_a)
    b = builder.action("action_b", action_b)
    c = builder.action("action_c", action_c)

    seq = builder.sequence(a, b, memory=True)
    root = builder.selector(seq, c, memory=False)

    tree = builder.build(root)

    # Serialize
    tree_dict = tree.to_dict()

    print("\n--- Serialized Tree (JSON) ---")
    print(json.dumps(tree_dict, indent=2))
    print("--- End of Serialization ---\n")

    # Save to file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(tree_dict, f, indent=2)
        print(f"Tree saved to: {f.name}")
    print()


# ============================================================================
# Demo 3: Visual Documentation
# ============================================================================


async def demo_documentation():
    """Generate documentation with embedded Mermaid diagrams."""
    print("\n" + "=" * 60)
    print("Demo 3: Documentation with Mermaid")
    print("=" * 60)

    builder = TreeBuilder("DeliveryBot")

    async def has_cargo(bb):
        return bb.value > 0

    async def has_battery(bb):
        return bb.value > 30

    async def deliver(bb):
        bb.value = 0
        return Status.SUCCESS

    async def recharge(bb):
        bb.value = 100
        return Status.SUCCESS

    async def load(bb):
        bb.value = 50
        return Status.SUCCESS

    cargo_check = builder.condition("has_cargo", has_cargo)
    battery_check = builder.condition("has_battery", has_battery)

    deliver_action = builder.action("deliver", deliver)
    recharge_action = builder.action("recharge", recharge)
    load_action = builder.action("load", load)

    # Tree: if has cargo -> deliver, else -> load
    # Either way, recharge if battery low
    delivery_seq = builder.sequence(cargo_check, deliver_action, memory=True)
    load_seq = builder.sequence(load_action, battery_check, memory=True)
    main_choice = builder.selector(delivery_seq, load_seq, memory=False)
    root = builder.sequence(main_choice, recharge_action, memory=True)

    tree = builder.build(root)

    print("Delivery Bot Behavior Tree")
    print("=" * 40)
    print(tree.to_mermaid())
    print("=" * 40)
    print()


# ============================================================================
# Demo 4: Run Tree and Compare Visualization
# ============================================================================


async def demo_run_and_compare():
    """Run a tree and compare execution with visualization."""
    print("\n" + "=" * 60)
    print("Demo 4: Run and Visualize")
    print("=" * 60)

    builder = TreeBuilder("CountingTree")

    counter = 0

    async def increment(bb):
        nonlocal counter
        counter += 1
        print(f"  Counter: {counter}")
        return Status.SUCCESS

    async def is_done(bb):
        return counter >= 3

    inc = builder.action("increment", increment)
    done_check = builder.condition("is_done", is_done)

    root = builder.sequence(inc, done_check, memory=True)
    tree = builder.build(root)

    print("\nTree structure:")
    print(tree.to_mermaid())
    print()

    print("Execution:")
    bb = SimpleBlackboard()
    runner = Runner(tree, bb=bb)

    for i in range(4):
        result = await runner.tick()
        print(f"Tick {i + 1}: {result}")
        if result != Status.RUNNING:
            break


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all demos."""
    await demo_mermaid_export()
    await demo_serialization()
    await demo_documentation()
    await demo_run_and_compare()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key Features Demonstrated:

1. Mermaid Diagram Export
   - Use tree.to_mermaid() to generate flowchart
   - Render at https://mermaid.live/
   - Embed in Markdown with ```mermaid blocks
   - Useful for documentation and code reviews

2. JSON Serialization
   - Use tree.to_dict() to get serializable representation
   - Save to files for version control
   - Share tree configurations
   - Load with function registry (not yet fully implemented)

Example Use Cases:

- Documentation: Generate diagrams for technical docs
- Code Review: Visualize tree structure in PRs
- Debugging: Compare expected vs actual structure
- Configuration: Save/load tree variants
- Testing: A/B test different tree structures
""")


if __name__ == "__main__":
    asyncio.run(main())
