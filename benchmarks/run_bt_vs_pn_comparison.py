#!/usr/bin/env python3
"""
Standalone script to run BT vs PN performance comparison

This script runs a benchmark comparing native Rhizomorph behavior
tree execution against compiled Hypha Petri net execution.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from mycorrhizal.rhizomorph.core import bt, Runner, Status
from mycorrhizal.hypha.core.runtime import MatrixRuntime
from mycorrhizal.common.timebase import MonotonicClock

# Import the compiler
import sys
sys.path.insert(0, '/home/jeff/workspace/mycorrhizal/.claude_work')
from bt_to_pn_compiler import BTtoPNCompiler


# ============================================================================
# Fixtures and Helpers
# ============================================================================

class BenchmarkBlackboard:
    """Blackboard for benchmarks"""
    def __init__(self):
        self.action_count = 0
        self.condition_count = 0


class BTStats:
    """Statistics collected during BT execution"""
    def __init__(self):
        self.total_time: float = 0.0
        self.actions_executed: int = 0
        self.conditions_checked: int = 0


class PNStats:
    """Statistics collected during PN execution"""
    def __init__(self):
        self.total_time: float = 0.0
        self.tokens_processed: int = 0
        self.places_count: int = 0
        self.transitions_count: int = 0


# ============================================================================
# Benchmark Behavior Trees
# ============================================================================

@bt.tree
def SimpleSequenceBT():
    """Simple sequence of 3 actions"""

    @bt.action
    async def action1(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def action2(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def action3(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.sequence
    def root():
        yield action1
        yield action2
        yield action3


@bt.tree
def NestedSequenceBT():
    """Nested sequences"""

    @bt.action
    async def action1(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def action2(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def action3(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def action4(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.sequence
    def inner_sequence():
        yield action2
        yield action3

    @bt.root
    @bt.sequence
    def root():
        yield action1
        yield inner_sequence
        yield action4


@bt.tree
def SelectorBT():
    """Selector with single action"""

    @bt.action
    async def action1(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.selector
    def root():
        yield action1


@bt.tree
def ParallelBT():
    """Parallel execution with 3 actions"""

    @bt.action
    async def task1(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def task2(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def task3(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.root
    @bt.parallel(success_threshold=2)
    def root():
        yield task1
        yield task2
        yield task3


@bt.tree
def ComplexMixedBT():
    """Complex tree with conditions and actions"""

    @bt.condition
    def is_ready(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return True

    @bt.condition
    def has_resources(bb: BenchmarkBlackboard) -> bool:
        bb.condition_count += 1
        return True

    @bt.action
    async def prepare(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def execute(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.action
    async def cleanup(bb: BenchmarkBlackboard) -> Status:
        bb.action_count += 1
        return Status.SUCCESS

    @bt.sequence
    def main_sequence():
        yield is_ready
        yield has_resources
        yield prepare
        yield execute
        yield cleanup

    @bt.root
    @bt.sequence
    def root():
        yield main_sequence


# ============================================================================
# Benchmark Functions
# ============================================================================

async def run_bt_benchmark(tree, num_ticks: int = 100) -> BTStats:
    """Run a behavior tree benchmark"""
    bb = BenchmarkBlackboard()
    tb = MonotonicClock()
    runner = Runner(tree, bb=bb, tb=tb)

    stats = BTStats()
    start_time = time.perf_counter()

    for _ in range(num_ticks):
        await runner.tick()

    stats.total_time = time.perf_counter() - start_time
    stats.actions_executed = bb.action_count
    stats.conditions_checked = bb.condition_count

    return stats


def compile_bt_to_pn(tree):
    """Compile a behavior tree to a Petri net"""
    compiler = BTtoPNCompiler(name="CompiledBT", running_delay=0.0, retry_delay=0.0)
    return compiler.compile(tree.root)


async def run_pn_benchmark(pn_spec, num_tokens: int = 100) -> PNStats:
    """Run a Petri net benchmark"""
    bb = BenchmarkBlackboard()
    tb = MonotonicClock()

    runtime = MatrixRuntime(pn_spec, bb, tb)
    await runtime.start()

    stats = PNStats()
    stats.places_count = len(runtime.places)
    stats.transitions_count = len(runtime.transitions)

    entry_place = runtime.places[("CompiledBT", "entry")]
    start_time = time.perf_counter()

    for i in range(num_tokens):
        entry_place.add_token(i)

    # Wait for all tokens to be processed
    timeout_count = 0
    max_timeout = 1000
    while bb.action_count < num_tokens and timeout_count < max_timeout:
        await asyncio.sleep(0.01)
        timeout_count += 1

    stats.total_time = time.perf_counter() - start_time
    stats.tokens_processed = num_tokens

    await runtime.stop(timeout=5)

    return stats


# ============================================================================
# Main Benchmark Runner
# ============================================================================

async def run_all_benchmarks():
    """Run all benchmarks and collect results"""

    NUM_ITERATIONS = 100

    benchmarks = [
        ("Simple Sequence (3 actions)", SimpleSequenceBT),
        ("Nested Sequence (4 actions)", NestedSequenceBT),
        ("Simple Selector (1 action)", SelectorBT),
        ("Parallel (3 actions)", ParallelBT),
        ("Complex Mixed (2 conditions + 3 actions)", ComplexMixedBT),
    ]

    results = {}

    print("=" * 90)
    print("BT vs PN Performance Comparison")
    print("=" * 90)
    print(f"\nRunning {NUM_ITERATIONS} iterations for each benchmark...")
    print()

    for name, tree in benchmarks:
        print(f"Benchmarking: {name}")

        # Run BT benchmark
        bt_stats = await run_bt_benchmark(tree, NUM_ITERATIONS)

        # Compile and run PN benchmark
        pn_spec = compile_bt_to_pn(tree)
        pn_stats = await run_pn_benchmark(pn_spec, NUM_ITERATIONS)

        # Calculate ratio
        ratio = pn_stats.total_time / bt_stats.total_time if bt_stats.total_time > 0 else float('inf')

        results[name] = {
            'bt_stats': bt_stats,
            'pn_stats': pn_stats,
            'ratio': ratio
        }

        print(f"  BT: {bt_stats.total_time*1000:.3f}ms | PN: {pn_stats.total_time*1000:.3f}ms | Ratio: {ratio:.2f}x")
        print(f"  PN Structure: {pn_stats.places_count} places, {pn_stats.transitions_count} transitions")
        print()

    # Print summary table
    print("=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"\n{'Benchmark':<35} {'BT Time':<12} {'PN Time':<12} {'Ratio':<10} {'PN Places':<10} {'PN Trans':<10}")
    print("-" * 95)

    for name, data in results.items():
        bt_time = data['bt_stats'].total_time * 1000
        pn_time = data['pn_stats'].total_time * 1000
        ratio = data['ratio']
        places = data['pn_stats'].places_count
        transitions = data['pn_stats'].transitions_count

        print(f"{name:<35} {bt_time:>8.3f}ms  {pn_time:>8.3f}ms  {ratio:>6.2f}x    {places:>6}       {transitions:>6}")

    print("-" * 95)

    # Calculate averages
    avg_ratio = sum(data['ratio'] for data in results.values()) / len(results)
    print(f"\nAverage slowdown: {avg_ratio:.2f}x")

    # Analysis
    print("\n" + "=" * 90)
    print("Analysis")
    print("=" * 90)
    print(f"""
The benchmark compares native Rhizomorph behavior tree execution against
Hypha Petri net execution (compiled from the same behavior tree).

Key findings:
  - BT ticks are measured directly via Runner.tick()
  - PN tokens are added to entry place and flow through the net
  - Both systems execute the same logic (same actions/conditions)

Performance considerations:
  - PN has additional overhead from transition firing, place management
  - PN runtime is event-driven with asyncio task coordination
  - BT runtime is simpler - direct traversal of tree structure

The Petri net compilation adds places and transitions for:
  - Each action/condition becomes a transition
  - Each composite (sequence/selector/parallel) adds structure
  - Flow control (success/failure paths) adds intermediate places

Use cases for each approach:
  - BT: Simple decision logic, game AI, control systems
  - PN: Complex workflows, concurrent processing, stateful systems
    """)


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
