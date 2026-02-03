#!/usr/bin/env python3
"""
Example: Programmatic Behavior Tree Builder

This demonstrates the TreeBuilder API for constructing behavior trees
without decorators, similar to how NetBuilder works in Hypha.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from mycorrhizal.rhizomorph.core import TreeBuilder, Runner, Status


@dataclass
class DemoBlackboard:
    """Shared state for the behavior tree"""
    battery: int = 100
    target_found: bool = False
    patrol_count: int = 0
    scan_count: int = 0
    engage_success: bool = False


# ============================================================================
# Demo 1: Basic Programmatic Tree
# ============================================================================

def build_basic_tree():
    """Build a simple behavior tree programmatically"""
    builder = TreeBuilder("BasicDemo")

    # Define conditions
    has_battery = builder.condition("has_battery", lambda bb: bb.battery > 20)
    target_visible = builder.condition("target_visible", lambda bb: bb.target_found)

    # Define actions
    async def scan_action(bb: DemoBlackboard) -> Status:
        bb.scan_count += 1
        print(f"  [Scan] Scanning for targets... (scan #{bb.scan_count})")
        if bb.scan_count >= 2:
            bb.target_found = True
            print(f"  [Scan] Target found!")
            return Status.SUCCESS
        return Status.RUNNING

    async def patrol_action(bb: DemoBlackboard) -> Status:
        bb.patrol_count += 1
        print(f"  [Patrol] Patrolling... (patrol #{bb.patrol_count})")
        if bb.patrol_count >= 3:
            return Status.SUCCESS
        return Status.RUNNING

    async def engage_action(bb: DemoBlackboard) -> Status:
        print(f"  [Engage] Engaging target!")
        bb.engage_success = True
        return Status.SUCCESS

    # Create composites
    scan = builder.action("scan", scan_action)
    patrol = builder.action("patrol", patrol_action)
    engage = builder.action("engage", engage_action)

    # Build tree structure:
    # Selector
    #   Sequence (engage if target visible and has battery)
    #     has_battery
    #     target_visible
    #     engage
    #   Sequence (patrol then scan)
    #     patrol
    #     scan
    engage_sequence = builder.sequence(has_battery, target_visible, engage, memory=True)

    patrol_scan_sequence = builder.sequence(patrol, scan, memory=True)

    root = builder.selector(engage_sequence, patrol_scan_sequence, memory=False)

    # Build final tree
    return builder.build(root)


async def demo_basic():
    """Run the basic programmatic tree demo"""
    print("=" * 60)
    print("Demo 1: Basic Programmatic Tree")
    print("=" * 60)

    bb = DemoBlackboard(battery=50)
    tree = build_basic_tree()

    runner = Runner(tree, bb=bb)

    print(f"\nInitial state: battery={bb.battery}, target_found={bb.target_found}")
    print("\nRunning tree:\n")

    result = await runner.tick_until_complete()

    print(f"\nFinal state: battery={bb.battery}, target_found={bb.target_found}")
    print(f"Result: {result}")
    print(f"Patrols: {bb.patrol_count}, Scans: {bb.scan_count}")


# ============================================================================
# Demo 2: Fluent Wrapper Chaining
# ============================================================================

def build_wrapped_tree():
    """Build a tree with fluent wrapper chains"""
    builder = TreeBuilder("WrappedDemo")

    async def risky_action(bb: DemoBlackboard) -> Status:
        import random
        print(f"  [RiskyAction] Attempting risky operation...")
        if random.random() > 0.5:
            print(f"  [RiskyAction] Success!")
            return Status.SUCCESS
        print(f"  [RiskyAction] Failed, will retry...")
        return Status.FAILURE

    # Chain wrappers: retry 3 times, with 1 second timeout, guarded by battery check
    battery_ok = builder.condition("battery_ok", lambda bb: bb.battery > 10)
    risky = builder.action("risky", risky_action)

    # Fluent chaining like decorators
    wrapped = (
        risky
        .gate(battery_ok)
        .retry(max_attempts=3)
        .timeout(seconds=1.0)
    )

    return builder.build(wrapped)


async def demo_wrapped():
    """Run the wrapper chaining demo"""
    print("\n" + "=" * 60)
    print("Demo 2: Fluent Wrapper Chaining")
    print("=" * 60)

    bb = DemoBlackboard(battery=50)
    tree = build_wrapped_tree()

    runner = Runner(tree, bb=bb)

    print(f"\nRunning tree with retry (3x), timeout (1s), and gate...\n")

    result = await runner.tick_until_complete()

    print(f"\nResult: {result}")


# ============================================================================
# Demo 3: Subtree Composition (Two Programmatic Trees)
# ============================================================================

def build_engage_subtree():
    """Build a reusable engage subtree"""
    builder = TreeBuilder("Engage")

    battery_ok = builder.condition("battery_ok", lambda bb: bb.battery > 10)
    target_visible = builder.condition("target_visible", lambda bb: bb.target_found)

    async def approach(bb: DemoBlackboard) -> Status:
        print(f"  [Approach] Approaching target...")
        return Status.SUCCESS

    async def attack(bb: DemoBlackboard) -> Status:
        print(f"  [Attack] Attacking target!")
        return Status.SUCCESS

    engage_seq = builder.sequence(
        battery_ok,
        target_visible,
        builder.action("approach", approach),
        builder.action("attack", attack),
        memory=True
    )

    return builder.build(engage_seq)


def build_composed_tree():
    """Build a tree that uses a subtree"""
    builder = TreeBuilder("ComposedDemo")

    # Get the engage subtree
    engage_tree = build_engage_subtree()

    async def scan(bb: DemoBlackboard) -> Status:
        bb.scan_count += 1
        print(f"  [Scan] Scanning... (scan #{bb.scan_count})")
        if bb.scan_count >= 1:
            bb.target_found = True
            print(f"  [Scan] Target found!")
            return Status.SUCCESS
        return Status.RUNNING

    async def retreat(bb: DemoBlackboard) -> Status:
        print(f"  [Retreat] Retreating to safety...")
        return Status.SUCCESS

    # Compose using the subtree
    engage_subtree = builder.subtree(engage_tree)
    scan_action = builder.action("scan", scan)
    retreat_action = builder.action("retreat", retreat)

    root = builder.selector(
        engage_subtree,
        builder.sequence(scan_action, memory=True),
        retreat_action,
        memory=False
    )

    return builder.build(root)


async def demo_composed():
    """Run the subtree composition demo"""
    print("\n" + "=" * 60)
    print("Demo 3: Subtree Composition")
    print("=" * 60)

    bb = DemoBlackboard(battery=50, target_found=False)
    tree = build_composed_tree()

    runner = Runner(tree, bb=bb)

    print(f"\nRunning composed tree with engage subtree...\n")

    result = await runner.tick_until_complete()

    print(f"\nResult: {result}")


# ============================================================================
# Demo 4: Parallel Execution
# ============================================================================

def build_parallel_tree():
    """Build a tree with parallel execution"""
    builder = TreeBuilder("ParallelDemo")

    async def task_a(bb: DemoBlackboard) -> Status:
        print(f"  [TaskA] Starting task A")
        await asyncio.sleep(0.1)
        print(f"  [TaskA] Completed task A")
        return Status.SUCCESS

    async def task_b(bb: DemoBlackboard) -> Status:
        print(f"  [TaskB] Starting task B")
        await asyncio.sleep(0.05)
        print(f"  [TaskB] Completed task B")
        return Status.SUCCESS

    async def task_c(bb: DemoBlackboard) -> Status:
        print(f"  [TaskC] Starting task C")
        await asyncio.sleep(0.15)
        print(f"  [TaskC] Completed task C")
        return Status.SUCCESS

    # Parallel: run all tasks, succeed when 2 complete, fail if 2 fail
    parallel_node = builder.parallel(
        builder.action("task_a", task_a),
        builder.action("task_b", task_b),
        builder.action("task_c", task_c),
        success_threshold=2,  # Succeed when 2 tasks complete
        failure_threshold=2,
    )

    return builder.build(parallel_node)


async def demo_parallel():
    """Run the parallel execution demo"""
    print("\n" + "=" * 60)
    print("Demo 4: Parallel Execution")
    print("=" * 60)

    bb = DemoBlackboard()
    tree = build_parallel_tree()

    runner = Runner(tree, bb=bb)

    print(f"\nRunning parallel tasks (success_threshold=2)...\n")

    result = await runner.tick_until_complete()

    print(f"\nResult: {result}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos"""
    await demo_basic()
    await demo_wrapped()
    await demo_composed()
    await demo_parallel()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
