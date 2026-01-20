#!/usr/bin/env python3
"""Simple benchmark runner for Septum FSM."""

import sys
sys.path.insert(0, "src")

import asyncio
import time
import gc
import tracemalloc
from enum import Enum, auto

from mycorrhizal.septum.core import (
    septum,
    StateMachine,
    LabeledTransition,
    StateConfiguration,
    SharedContext,
)


# ============================================================================
# Benchmark States
# ============================================================================


@septum.state()
def BenchmarkState():
    """Simple state for benchmarking."""

    @septum.events
    class Events(Enum):
        NEXT = auto()

    @septum.on_state
    async def on_state(ctx: SharedContext):
        return Events.NEXT

    @septum.transitions
    def transitions():
        return [
            LabeledTransition(Events.NEXT, BenchmarkState),
        ]


# ============================================================================
# Benchmarks
# ============================================================================


async def benchmark_state_transition_overhead():
    """Measure time per state transition."""
    print("\n" + "=" * 70)
    print("Benchmark 1: State Transition Overhead")
    print("=" * 70)

    fsm = StateMachine(initial_state=BenchmarkState)
    await fsm.initialize()

    # Warm up
    for _ in range(100):
        await fsm.tick()

    # Benchmark
    iterations = 10000
    start = time.perf_counter()

    for _ in range(iterations):
        await fsm.tick()

    end = time.perf_counter()
    total_time = end - start

    per_transition_us = (total_time / iterations) * 1_000_000

    print(f"State transition overhead: {per_transition_us:.2f} µs per transition")
    print(f"  Total time for {iterations} transitions: {total_time:.4f}s")
    print(f"  Throughput: {iterations / total_time:.0f} transitions/sec")

    return per_transition_us


async def benchmark_fsm_creation_overhead():
    """Measure time to create and initialize FSM instances."""
    print("\n" + "=" * 70)
    print("Benchmark 2: FSM Creation Overhead")
    print("=" * 70)

    iterations = 1000

    start = time.perf_counter()

    for _ in range(iterations):
        fsm = StateMachine(initial_state=BenchmarkState)
        await fsm.initialize()

    end = time.perf_counter()
    total_time = end - start

    per_fsm_ms = (total_time / iterations) * 1000

    print(f"FSM creation overhead: {per_fsm_ms:.3f} ms per FSM")
    print(f"  Total time for {iterations} FSMs: {total_time:.4f}s")
    print(f"  Throughput: {iterations / total_time:.0f} FSMs/sec")

    return per_fsm_ms


async def benchmark_memory_usage():
    """Measure memory per FSM instance."""
    print("\n" + "=" * 70)
    print("Benchmark 3: Memory Usage")
    print("=" * 70)

    gc.collect()
    tracemalloc.start()

    # Create baseline
    snapshot1 = tracemalloc.take_snapshot()

    # Create 1000 FSMs
    fsms = []
    for _ in range(1000):
        fsm = StateMachine(initial_state=BenchmarkState)
        await fsm.initialize()
        fsms.append(fsm)

    snapshot2 = tracemalloc.take_snapshot()

    # Calculate memory
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    total_memory = sum(stat.size_diff for stat in top_stats)
    per_fsm_kb = total_memory / 1000 / 1024

    print(f"Memory usage: {per_fsm_kb:.2f} KB per FSM instance")
    print(f"  Total memory for 1000 FSMs: {total_memory / 1024:.2f} KB")

    tracemalloc.stop()

    return per_fsm_kb


async def benchmark_concurrent_fsm_performance():
    """Measure performance of running FSMs concurrently."""
    print("\n" + "=" * 70)
    print("Benchmark 4: Concurrent FSM Performance")
    print("=" * 70)

    async def run_fsm(fsm_id: int):
        """Run FSM for a few ticks."""
        fsm = StateMachine(initial_state=BenchmarkState)
        await fsm.initialize()

        for _ in range(100):
            await fsm.tick()

        return fsm_id

    # Test with different concurrency levels
    for num_fsms in [10, 50, 100]:
        start = time.perf_counter()

        tasks = [run_fsm(i) for i in range(num_fsms)]
        await asyncio.gather(*tasks)

        end = time.perf_counter()
        total_time = end - start

        print(f"  {num_fsms} concurrent FSMs: {total_time:.4f}s total")
        print(f"    Average per FSM: {(total_time / num_fsms) * 1000:.3f} ms")
        print(f"    Throughput: {num_fsms / total_time:.0f} FSMs/sec")


# ============================================================================
# Main
# ============================================================================


async def main():
    """Run all benchmarks and print summary."""
    print("\n" + "=" * 70)
    print("SEPTUM FSM PERFORMANCE BENCHMARKS")
    print("=" * 70)

    results = {}

    # Benchmark 1: State transition overhead
    result = await benchmark_state_transition_overhead()
    results["transition_overhead_us"] = result

    # Benchmark 2: FSM creation overhead
    result = await benchmark_fsm_creation_overhead()
    results["fsm_creation_ms"] = result

    # Benchmark 3: Memory usage
    result = await benchmark_memory_usage()
    results["memory_per_fsm_kb"] = result

    # Benchmark 4: Concurrent FSM performance
    await benchmark_concurrent_fsm_performance()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"State transition overhead: {results['transition_overhead_us']:.2f} µs/transition")
    print(f"FSM creation: {results['fsm_creation_ms']:.3f} ms/instance")
    print(f"Memory per FSM: {results['memory_per_fsm_kb']:.2f} KB/instance")
    print("=" * 70)
    print("\nPractical Limits Documented:")
    print(f"  - State transition overhead: {results['transition_overhead_us']:.2f} µs/transition")
    print(f"    → Can handle ~{1_000_000 / results['transition_overhead_us']:.0f} transitions/second")
    print(f"  - Memory usage: {results['memory_per_fsm_kb']:.2f} KB/instance")
    print(f"    → 10,000 FSMs would use ~{results['memory_per_fsm_kb'] * 10000 / 1024:.1f} MB")
    print(f"  - FSM creation: {results['fsm_creation_ms']:.3f} ms/instance")
    print(f"    → Can create ~{1000 / results['fsm_creation_ms']:.0f} FSMs/second")
    print("=" * 70)
    print("\nAll benchmarks completed successfully!")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    asyncio.run(main())
