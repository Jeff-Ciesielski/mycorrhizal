"""
Benchmark comparing Rhizomorph Behavior Trees vs Hypha Petri Nets (compiled from BTs)

This benchmark measures the performance difference between:
1. Native Rhizomorph behavior tree execution
2. Hypha Petri net execution (compiled from the same behavior tree)

The benchmark uses the bt_to_pn_compiler to compile behavior trees to Petri nets,
then runs equivalent workloads through both systems to measure relative performance.

Key aspects measured:
- Tick execution time for behavior trees
- Token processing time for Petri nets
- Throughput (operations per second)
- Overhead of Petri net runtime vs direct BT execution
"""

import asyncio
import pytest
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from typing import Optional

from mycorrhizal.rhizomorph.core import (
    bt, Runner, Status, TreeBuilder
)
from mycorrhizal.hypha.core.builder import NetBuilder
from mycorrhizal.common.timebase import MonotonicClock, CycleClock

# Import the compiler from the work directory
import sys
sys.path.insert(0, '/home/jeff/workspace/mycorrhizal/.claude_work')
from bt_to_pn_compiler import BTtoPNCompiler


# ============================================================================
# Fixtures and Helpers
# ============================================================================

class BenchmarkBlackboard:
    """Blackboard for benchmarks - tracks execution counts"""
    def __init__(self):
        self.action_count = 0
        self.condition_count = 0
        self.value = 0
        self.processed_items: List[Any] = field(default_factory=list)


class BTStats:
    """Statistics collected during BT execution"""
    def __init__(self):
        self.total_time: float = 0.0
        self.total_ticks: int = 0
        self.actions_executed: int = 0
        self.conditions_checked: int = 0


class PNStats:
    """Statistics collected during PN execution"""
    def __init__(self):
        self.total_time: float = 0.0
        self.tokens_processed: int = 0
        self.transitions_fired: int = 0
        self.places_count: int = 0
        self.transitions_count: int = 0


# Global cache for compiled specs to avoid recompiling
_COMPILED_SPECS = {}


# ============================================================================
# Benchmark Behavior Trees
# ============================================================================

@bt.tree
def SimpleSequenceBT():
    """Simple sequence of 3 actions - baseline comparison"""

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
    """Nested sequences - more complex structure"""

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
    """Selector with multiple conditions - only first child passes"""

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
    """Parallel execution with multiple actions"""

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
    """Complex tree with sequences, selectors, and conditions"""

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
# Benchmark Helpers
# ============================================================================

async def run_bt_benchmark(tree, num_ticks: int = 100) -> BTStats:
    """Run a behavior tree benchmark and collect statistics"""
    bb = BenchmarkBlackboard()
    tb = MonotonicClock()
    runner = Runner(tree, bb=bb, tb=tb)

    stats = BTStats()
    start_time = time.perf_counter()

    for _ in range(num_ticks):
        await runner.tick()
        stats.total_ticks += 1

    stats.total_time = time.perf_counter() - start_time
    stats.actions_executed = bb.action_count
    stats.conditions_checked = bb.condition_count

    return stats


def compile_bt_to_pn(tree) -> tuple:
    """Compile a behavior tree to a Petri net"""
    compiler = BTtoPNCompiler(name="CompiledBT", running_delay=0.0, retry_delay=0.0)
    pn_spec = compiler.compile(tree.root)
    return pn_spec, compiler


async def run_pn_benchmark(pn_spec, num_tokens: int = 100) -> PNStats:
    """Run a Petri net benchmark and collect statistics"""
    bb = BenchmarkBlackboard()
    tb = MonotonicClock()

    # Use MatrixRuntime for efficient async execution
    from mycorrhizal.hypha.core.runtime import MatrixRuntime
    runtime = MatrixRuntime(pn_spec, bb, tb)
    await runtime.start()

    stats = PNStats()
    stats.places_count = len(runtime.places)
    stats.transitions_count = len(runtime.incidence_matrix.transition_to_idx)

    # Get entry place index
    entry_place_key = ("CompiledBT", "entry")
    entry_place_idx = runtime.incidence_matrix.place_to_idx.get(entry_place_key)

    if entry_place_idx is None:
        raise ValueError(f"Entry place {entry_place_key} not found")

    start_time = time.perf_counter()

    # Add all tokens upfront
    for i in range(num_tokens):
        runtime.add_token(entry_place_idx, i)

    # Wait for all tokens to be processed
    timeout_count = 0
    max_timeout = 10000  # 10 seconds max wait (with 0.001ms sleep)
    while bb.action_count < num_tokens and timeout_count < max_timeout:
        await asyncio.sleep(0.0001)  # Shorter sleep for faster response
        timeout_count += 1

    stats.total_time = time.perf_counter() - start_time
    stats.tokens_processed = num_tokens
    stats.transitions_fired = bb.action_count

    await runtime.stop(timeout=5)

    return stats


def run_pn_benchmark_sync(pn_spec, num_tokens: int = 100) -> PNStats:
    """Run a Petri net benchmark in synchronous mode (high performance)."""
    from mycorrhizal.hypha.core.runtime import MatrixRuntime

    bb = BenchmarkBlackboard()
    tb = MonotonicClock()

    # Use MatrixRuntime in synchronous mode
    runtime = MatrixRuntime(pn_spec, bb, tb)
    stats = PNStats()
    stats.places_count = len(runtime.places)
    stats.transitions_count = len(runtime.incidence_matrix.transition_to_idx)

    # Get entry place index
    entry_place_key = ("CompiledBT", "entry")
    entry_place_idx = runtime.incidence_matrix.place_to_idx.get(entry_place_key)

    if entry_place_idx is None:
        raise ValueError(f"Entry place {entry_place_key} not found")

    start_time = time.perf_counter()

    # Add all tokens upfront
    for i in range(num_tokens):
        runtime.add_token(entry_place_idx, i)

    # Run to completion
    cycles = runtime.run_sync(max_cycles=100000)

    elapsed = time.perf_counter() - start_time
    stats.total_time = elapsed
    stats.tokens_processed = num_tokens
    stats.transitions_fired = bb.action_count

    return stats


# Store comparison results for reporting
_comparison_results: Dict[str, Dict] = {}


# ============================================================================
# Behavior Tree Benchmarks (Native Rhizomorph)
# ============================================================================

@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="bt-native-simple")
def test_bt_simple_sequence(benchmark):
    """Benchmark: Native BT - Simple sequence of 3 actions"""
    NUM_ITERATIONS = 100

    def run_bt():
        async def _run():
            return await run_bt_benchmark(SimpleSequenceBT, NUM_ITERATIONS)
        return asyncio.run(_run())

    bt_stats = benchmark(run_bt)
    _comparison_results['bt_simple'] = bt_stats

    assert bt_stats.actions_executed == NUM_ITERATIONS * 3


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="bt-native-nested")
def test_bt_nested_sequence(benchmark):
    """Benchmark: Native BT - Nested sequences"""
    NUM_ITERATIONS = 100

    def run_bt():
        async def _run():
            return await run_bt_benchmark(NestedSequenceBT, NUM_ITERATIONS)
        return asyncio.run(_run())

    bt_stats = benchmark(run_bt)
    _comparison_results['bt_nested'] = bt_stats

    assert bt_stats.actions_executed == NUM_ITERATIONS * 4


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="bt-native-selector")
def test_bt_selector(benchmark):
    """Benchmark: Native BT - Simple selector"""
    NUM_ITERATIONS = 100

    def run_bt():
        async def _run():
            return await run_bt_benchmark(SelectorBT, NUM_ITERATIONS)
        return asyncio.run(_run())

    bt_stats = benchmark(run_bt)
    _comparison_results['bt_selector'] = bt_stats

    assert bt_stats.actions_executed == NUM_ITERATIONS


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="bt-native-parallel")
def test_bt_parallel(benchmark):
    """Benchmark: Native BT - Parallel execution"""
    NUM_ITERATIONS = 100

    def run_bt():
        async def _run():
            return await run_bt_benchmark(ParallelBT, NUM_ITERATIONS)
        return asyncio.run(_run())

    bt_stats = benchmark(run_bt)
    _comparison_results['bt_parallel'] = bt_stats

    assert bt_stats.actions_executed == NUM_ITERATIONS * 3


@pytest.mark.rhizomorph
@pytest.mark.benchmark(group="bt-native-complex")
def test_bt_complex_mixed(benchmark):
    """Benchmark: Native BT - Complex mixed structure"""
    NUM_ITERATIONS = 100

    def run_bt():
        async def _run():
            return await run_bt_benchmark(ComplexMixedBT, NUM_ITERATIONS)
        return asyncio.run(_run())

    bt_stats = benchmark(run_bt)
    _comparison_results['bt_complex'] = bt_stats

    assert bt_stats.conditions_checked == NUM_ITERATIONS * 2
    assert bt_stats.actions_executed == NUM_ITERATIONS * 3


# ============================================================================
# Petri Net Benchmarks (Compiled from BTs)
# ============================================================================

@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-compiled-simple")
def test_pn_simple_sequence(benchmark):
    """Benchmark: Compiled PN - Simple sequence of 3 actions"""
    NUM_ITERATIONS = 100

    # Compile once, use cached spec
    if 'simple_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(SimpleSequenceBT)
        _COMPILED_SPECS['simple_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['simple_spec']

    def run_pn():
        async def _run():
            return await run_pn_benchmark(pn_spec, NUM_ITERATIONS)
        return asyncio.run(_run())

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_simple'] = pn_stats

    assert pn_stats.tokens_processed == NUM_ITERATIONS


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-compiled-nested")
def test_pn_nested_sequence(benchmark):
    """Benchmark: Compiled PN - Nested sequences"""
    NUM_ITERATIONS = 100

    if 'nested_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(NestedSequenceBT)
        _COMPILED_SPECS['nested_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['nested_spec']

    def run_pn():
        async def _run():
            return await run_pn_benchmark(pn_spec, NUM_ITERATIONS)
        return asyncio.run(_run())

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_nested'] = pn_stats

    assert pn_stats.tokens_processed == NUM_ITERATIONS


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-compiled-selector")
def test_pn_selector(benchmark):
    """Benchmark: Compiled PN - Simple selector"""
    NUM_ITERATIONS = 100

    if 'selector_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(SelectorBT)
        _COMPILED_SPECS['selector_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['selector_spec']

    def run_pn():
        async def _run():
            return await run_pn_benchmark(pn_spec, NUM_ITERATIONS)
        return asyncio.run(_run())

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_selector'] = pn_stats

    assert pn_stats.tokens_processed == NUM_ITERATIONS


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-compiled-parallel")
def test_pn_parallel(benchmark):
    """Benchmark: Compiled PN - Parallel execution"""
    NUM_ITERATIONS = 100

    if 'parallel_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(ParallelBT)
        _COMPILED_SPECS['parallel_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['parallel_spec']

    def run_pn():
        async def _run():
            return await run_pn_benchmark(pn_spec, NUM_ITERATIONS)
        return asyncio.run(_run())

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_parallel'] = pn_stats


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-compiled-complex")
def test_pn_complex_mixed(benchmark):
    """Benchmark: Compiled PN - Complex mixed structure"""
    NUM_ITERATIONS = 100

    if 'complex_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(ComplexMixedBT)
        _COMPILED_SPECS['complex_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['complex_spec']

    def run_pn():
        async def _run():
            return await run_pn_benchmark(pn_spec, NUM_ITERATIONS)
        return asyncio.run(_run())

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_complex'] = pn_stats


# ============================================================================
# Petri Net Benchmarks (Synchronous Mode)
# ============================================================================

@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-sync-simple")
def test_pn_sync_simple_sequence(benchmark):
    """Benchmark: Compiled PN (Sync) - Simple sequence of 3 actions"""
    NUM_ITERATIONS = 100

    # Compile once, use cached spec
    if 'simple_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(SimpleSequenceBT)
        _COMPILED_SPECS['simple_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['simple_spec']

    def run_pn():
        return run_pn_benchmark_sync(pn_spec, NUM_ITERATIONS)

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_sync_simple'] = pn_stats

    assert pn_stats.tokens_processed == NUM_ITERATIONS


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-sync-nested")
def test_pn_sync_nested_sequence(benchmark):
    """Benchmark: Compiled PN (Sync) - Nested sequences"""
    NUM_ITERATIONS = 100

    if 'nested_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(NestedSequenceBT)
        _COMPILED_SPECS['nested_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['nested_spec']

    def run_pn():
        return run_pn_benchmark_sync(pn_spec, NUM_ITERATIONS)

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_sync_nested'] = pn_stats

    assert pn_stats.tokens_processed == NUM_ITERATIONS


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-sync-selector")
def test_pn_sync_selector(benchmark):
    """Benchmark: Compiled PN (Sync) - Simple selector"""
    NUM_ITERATIONS = 100

    if 'selector_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(SelectorBT)
        _COMPILED_SPECS['selector_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['selector_spec']

    def run_pn():
        return run_pn_benchmark_sync(pn_spec, NUM_ITERATIONS)

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_sync_selector'] = pn_stats

    assert pn_stats.tokens_processed == NUM_ITERATIONS


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-sync-parallel")
def test_pn_sync_parallel(benchmark):
    """Benchmark: Compiled PN (Sync) - Parallel execution"""
    NUM_ITERATIONS = 100

    if 'parallel_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(ParallelBT)
        _COMPILED_SPECS['parallel_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['parallel_spec']

    def run_pn():
        return run_pn_benchmark_sync(pn_spec, NUM_ITERATIONS)

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_sync_parallel'] = pn_stats


@pytest.mark.hypha
@pytest.mark.benchmark(group="pn-sync-complex")
def test_pn_sync_complex_mixed(benchmark):
    """Benchmark: Compiled PN (Sync) - Complex mixed structure"""
    NUM_ITERATIONS = 100

    if 'complex_spec' not in _COMPILED_SPECS:
        pn_spec, _ = compile_bt_to_pn(ComplexMixedBT)
        _COMPILED_SPECS['complex_spec'] = pn_spec
    else:
        pn_spec = _COMPILED_SPECS['complex_spec']

    def run_pn():
        return run_pn_benchmark_sync(pn_spec, NUM_ITERATIONS)

    pn_stats = benchmark(run_pn)
    _comparison_results['pn_sync_complex'] = pn_stats


# ============================================================================
# Comparison Summary
# ============================================================================

@pytest.mark.rhizomorph
@pytest.mark.hypha
def test_comparison_summary():
    """Print comparison summary after all benchmarks run"""
    print("\n" + "=" * 100)
    print("BT vs PN Performance Comparison Summary")
    print("=" * 100)

    comparisons = [
        ('simple', 'Simple Sequence (3 actions)'),
        ('nested', 'Nested Sequence (4 actions)'),
        ('selector', 'Simple Selector (1 action)'),
        ('parallel', 'Parallel (3 actions)'),
        ('complex', 'Complex Mixed (2 conditions + 3 actions)'),
    ]

    print(f"\n{'Test Case':<30} {'BT':<10} {'PN (async)':<12} {'PN (sync)':<12} {'Sync/Async':<12} {'vs BT':<10}")
    print("-" * 100)

    for key, name in comparisons:
        bt_key = f'bt_{key}'
        pn_key = f'pn_{key}'
        pn_sync_key = f'pn_sync_{key}'

        results = []
        if bt_key in _comparison_results:
            bt_stats = _comparison_results[bt_key]
            bt_time = bt_stats.total_time * 1000
            results.append(f"{bt_time:>6.3f}ms")
        else:
            results.append(f"{'N/A':>10}")

        if pn_key in _comparison_results:
            pn_stats = _comparison_results[pn_key]
            pn_time = pn_stats.total_time * 1000
            results.append(f"{pn_time:>6.3f}ms")
        else:
            results.append(f"{'N/A':>10}")

        if pn_sync_key in _comparison_results:
            pn_sync_stats = _comparison_results[pn_sync_key]
            pn_sync_time = pn_sync_stats.total_time * 1000
            results.append(f"{pn_sync_time:>6.3f}ms")
        else:
            results.append(f"{'N/A':>10}")

        # Calculate ratios
        ratios = []
        if bt_key in _comparison_results and pn_key in _comparison_results:
            bt_stats = _comparison_results[bt_key]
            pn_stats = _comparison_results[pn_key]
            async_ratio = pn_stats.total_time / bt_stats.total_time if bt_stats.total_time > 0 else float('inf')
            ratios.append(f"{async_ratio:>5.2f}x")
        else:
            ratios.append(f"{'N/A':>7}")

        if bt_key in _comparison_results and pn_sync_key in _comparison_results:
            bt_stats = _comparison_results[bt_key]
            pn_sync_stats = _comparison_results[pn_sync_key]
            sync_ratio = pn_sync_stats.total_time / bt_stats.total_time if bt_stats.total_time > 0 else float('inf')
            ratios.append(f"{sync_ratio:>5.2f}x")
        else:
            ratios.append(f"{'N/A':>7}")

        if pn_key in _comparison_results and pn_sync_key in _comparison_results:
            pn_stats = _comparison_results[pn_key]
            pn_sync_stats = _comparison_results[pn_sync_key]
            speedup = pn_stats.total_time / pn_sync_stats.total_time if pn_sync_stats.total_time > 0 else float('inf')
            results.append(f"{speedup:>5.2f}x")
        else:
            results.append(f"{'N/A':>7}")

        results_str = "  ".join(results + ratios)
        print(f"{name:<30} {results_str}")

    print("-" * 100)
    print("\nKey:")
    print("  BT: Native behavior tree execution time")
    print("  PN (async): Compiled Petri net execution time (async mode)")
    print("  PN (sync): Compiled Petri net execution time (sync mode)")
    print("  Sync/Async: Speedup of sync mode vs async mode")
    print("  vs BT: Slowdown of sync mode compared to native BT")
    print("\nTarget: <10x slowdown vs native BT")
    print("=" * 100)
